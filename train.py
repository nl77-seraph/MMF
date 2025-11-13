"""
Base Class训练主程序
实现基于Epoch的多标签网站指纹识别训练
特别处理严重的类别不均衡问题（3:57正负样本比例）
支持单GPU和多GPU分布式训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.meta_traffic_dataloader import MetaTrafficDataLoader
from models.feature_extractors import MultiMetaFingerNet
from utils.metrics import MultiLabelMetrics
from utils.loss_functions import WeightedBCELoss, FocalLoss
from utils.model_manager import ModelManager


def setup_distributed_training(rank, world_size, config):
    """
    设置分布式训练环境
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        config: 配置字典
    """
    # 设置CUDA设备
    torch.cuda.set_device(rank)
    
    # 初始化进程组
    os.environ['MASTER_ADDR'] = config.get('master_addr', 'localhost')
    os.environ['MASTER_PORT'] = config.get('master_port', '12355')
    
    dist.init_process_group(
        backend=config.get('dist_backend', 'nccl'),
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    print(f"🚀 分布式训练初始化完成 - Rank: {rank}/{world_size}")


def cleanup_distributed_training():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """检查是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


class BaseClassTrainer:
    """
    Base Class训练器
    专门处理多标签分类的训练，重点解决类别不均衡问题
    支持单GPU和多GPU分布式训练
    """
    
    def __init__(self, config, rank=None, world_size=None):
        self.config = config
        self.rank = rank if rank is not None else 0
        self.world_size = world_size if world_size is not None else 1
        self.is_distributed = world_size is not None and world_size > 1
        
        # 设置设备
        if self.is_distributed:
            self.device = torch.device(f'cuda:{rank}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化训练组件
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_map = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # 日志和可视化（仅主进程）
        self.writer = None
        self.model_manager = None
        
        # 类别不均衡处理
        self.pos_weight = None
        self.class_weights = None
        
        if is_main_process():
            print(f"🚀 BaseClassTrainer初始化")
            print(f"  - 模式: {'分布式训练' if self.is_distributed else '单GPU训练'}")
            if self.is_distributed:
                print(f"  - Rank: {self.rank}/{self.world_size}")
            print(f"  - 设备: {self.device}")
            print(f"  - 类别数: {config['num_classes']}")
            print(f"  - 正负样本比例: ~{config['positive_ratio']}:1 (严重不均衡)")
    
    def setup_data_loaders(self):
        """设置数据加载器（支持分布式采样）"""
        if is_main_process():
            print("\n📦 设置数据加载器...")
        
        # 训练数据加载器（随机采样模式）
        train_loader_base = MetaTrafficDataLoader(
            query_json_path=self.config['train_query_json'],
            query_files_dir=self.config['train_query_dir'],
            support_root_dir=self.config['support_root_dir'],
            activated_classes=list(range(self.config['num_classes'])),
            target_length=self.config['sequence_length'],
            shots_per_class=self.config['shots_per_class'],
            batch_size=self.config['batch_size'],
            shuffle=not self.is_distributed,  # 分布式模式下由DistributedSampler控制
            num_workers=self.config['num_workers'],
            random_sampling=True  # 训练使用随机采样
        )
        
        # 如果是分布式训练，包装数据加载器
        if self.is_distributed:
            # 分布式采样器
            self.train_sampler = DistributedSampler(
                train_loader_base.query_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            
            # 重新创建DataLoader with DistributedSampler
            from torch.utils.data import DataLoader
            train_loader_base.query_loader = DataLoader(
                train_loader_base.query_dataset,
                batch_size=self.config['batch_size'],
                sampler=self.train_sampler,
                num_workers=self.config['num_workers'],
                collate_fn=train_loader_base._query_collate_fn
            )
        else:
            self.train_sampler = None
        
        self.train_loader = train_loader_base
        
        # 验证数据加载器（固定采样模式）
        # 验证数据不需要分布式采样，每个进程验证相同的数据
        self.val_loader = MetaTrafficDataLoader(
            query_json_path=self.config['val_query_json'],
            query_files_dir=self.config['val_query_dir'],
            support_root_dir=self.config['support_root_dir'],
            activated_classes=list(range(self.config['num_classes'])),
            target_length=self.config['sequence_length'],
            shots_per_class=self.config['shots_per_class'],
            batch_size=self.config['val_batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            random_sampling=True  # 验证使用固定采样
        )
        
        if is_main_process():
            print(f"  ✅ 训练集: {len(self.train_loader)} batches")
            print(f"  ✅ 验证集: {len(self.val_loader)} batches")
            if self.is_distributed:
                print(f"  ✅ 分布式采样器: 已启用")
    
    def setup_model(self):
        """设置模型（支持DDP包装）"""
        if is_main_process():
            print("\n🧠 设置网络模型...")
        
        self.model = MultiMetaFingerNet(
            num_classes=self.config['num_classes'],
            dropout=self.config['dropout'],
            support_blocks=self.config['support_blocks'],
            classification_method=self.config['classification_method'],
            unified_threshold=self.config['unified_threshold']
        ).to(self.device)
        
        # 如果是分布式训练，包装为DDP
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False  # 提升性能
            )
            if is_main_process():
                print(f"  ✅ DDP模型包装完成 - Device: {self.rank}")
        
        # 计算模型参数量（仅主进程输出）
        if is_main_process():
            model_for_count = self.model.module if self.is_distributed else self.model
            total_params = sum(p.numel() for p in model_for_count.parameters())
            trainable_params = sum(p.numel() for p in model_for_count.parameters() if p.requires_grad)
            
            print(f"  ✅ 模型参数: {total_params:,} 总量, {trainable_params:,} 可训练")
            print(f"  ✅ 模型已移至: {self.device}")
    
    def setup_loss_function(self):
        """设置损失函数，重点处理类别不均衡"""
        if is_main_process():
            print("\n⚖️ 设置损失函数（处理类别不均衡）...")
        
        # 计算正负样本权重
        positive_ratio = self.config['positive_ratio']  # 3:57的正负比例
        pos_weight = torch.tensor([positive_ratio] * self.config['num_classes']).to(self.device)
        self.pos_weight = pos_weight
        
        loss_type = self.config['loss_type']
        
        if loss_type == 'weighted_bce':
            self.criterion = WeightedBCELoss(pos_weight=pos_weight)
            if is_main_process():
                print(f"  ✅ 使用Weighted BCE Loss, 正样本权重: {positive_ratio:.1f}")
                
        elif loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config['focal_alpha'],
                gamma=self.config['focal_gamma'],
                pos_weight=pos_weight
            )
            if is_main_process():
                print(f"  ✅ 使用Focal Loss, alpha={self.config['focal_alpha']}, gamma={self.config['focal_gamma']}")
                
        else:
            # 标准BCE作为基准
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if is_main_process():
                print(f"  ✅ 使用标准BCE Loss with pos_weight")
    
    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        if is_main_process():
            print("\n🎯 设置优化器...")
        
        # 获取模型参数（考虑DDP包装）
        model_params = self.model.module.parameters() if self.is_distributed else self.model.parameters()
        
        # 创建优化器
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                model_params,
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                model_params,
                lr=self.config['learning_rate'],
                momentum=self.config['momentum'],
                weight_decay=self.config['weight_decay']
            )
        
        # 学习率调度器
        if self.config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=self.config['min_lr']
            )
        elif self.config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['step_size'],
                gamma=self.config['gamma']
            )
        
        if is_main_process():
            print(f"  ✅ 优化器: {self.config['optimizer']}")
            print(f"  ✅ 学习率: {self.config['learning_rate']}")
            print(f"  ✅ 调度器: {self.config['scheduler']}")
    
    def setup_logging(self):
        """设置日志和模型管理（仅主进程）"""
        if not is_main_process():
            return
            
        print("\n📊 设置日志系统...")
        
        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_ddp" if self.is_distributed else "_single"
        exp_name = f"base_training_{timestamp}{mode_suffix}"
        self.exp_dir = os.path.join(self.config['output_dir'], exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # TensorBoard
        log_dir = os.path.join(self.exp_dir, 'logs')
        self.writer = SummaryWriter(log_dir)
        
        # 模型管理器
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.model_manager = ModelManager(checkpoint_dir)
        
        print(f"  ✅ 实验目录: {self.exp_dir}")
        print(f"  ✅ TensorBoard日志: {log_dir}")
        print(f"  ✅ Checkpoint目录: {checkpoint_dir}")
    
    def train_epoch(self, epoch):
        """训练一个epoch（支持分布式训练）"""
        self.model.train()
        
        # 分布式训练需要设置epoch用于shuffle
        if self.is_distributed and self.train_sampler:
            self.train_sampler.set_epoch(epoch)
        
        train_losses = []
        all_train_logits = []
        all_train_labels = []
        
        # 训练循环
        for batch_idx, batch in enumerate(self.train_loader):
            query_data, support_data, support_masks, batch_info = batch
            
            # 数据移到设备
            query_data = query_data.to(self.device)
            support_data = support_data.to(self.device)
            support_masks = support_masks.to(self.device)
            query_labels = batch_info['query_labels'].to(self.device)
            
            # 前向传播
            results = self.model(query_data, support_data, support_masks)
            
            # 计算损失
            loss = self.criterion(results['logits'], query_labels.float())
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('grad_clip', 0) > 0:
                model_params = self.model.module.parameters() if self.is_distributed else self.model.parameters()
                torch.nn.utils.clip_grad_norm_(model_params, self.config['grad_clip'])
            
            self.optimizer.step()
            
            # 记录损失和预测（用于训练epoch评估）
            train_losses.append(loss.item())
            all_train_logits.append(results['logits'].detach().cpu())
            all_train_labels.append(query_labels.detach().cpu())
            
            # 在训练的10%时显示预测样本（仅主进程）
            if is_main_process() and batch_idx == len(self.train_loader) // 10:
                self.print_prediction_samples(results['logits'], query_labels, epoch)
            
            # 记录批级别指标到TensorBoard（仅主进程）
            if is_main_process() and batch_idx % 10 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], step)
                
                # 计算梯度范数
                model_params = self.model.module.parameters() if self.is_distributed else self.model.parameters()
                grad_norm = sum(p.grad.norm().item() ** 2 for p in model_params if p.grad is not None) ** 0.5
                self.writer.add_scalar('Train/GradientNorm', grad_norm, step)
        
        # 计算平均训练损失
        avg_train_loss = np.mean(train_losses)
        
        # 计算训练epoch的整体评估指标
        all_train_logits = torch.cat(all_train_logits, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        train_metrics = self.evaluate_training_epoch(all_train_logits, all_train_labels)
        
        # 记录训练指标到TensorBoard（仅主进程）
        if is_main_process():
            self.log_training_metrics(train_metrics, avg_train_loss, epoch)
        
        return avg_train_loss, train_metrics
    
    def evaluate_training_epoch(self, all_logits, all_labels):
        """对整个训练epoch进行评估"""
        return MultiLabelMetrics.compute_metrics(all_logits, all_labels)
    
    def log_training_metrics(self, train_metrics, train_loss, epoch):
        """记录训练指标到TensorBoard（仅主进程）"""
        if not is_main_process():
            return
            
        # 记录训练损失和指标（使用'Train/'前缀）
        self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        self.writer.add_scalar('Train/mAP', train_metrics['mAP'], epoch)
        self.writer.add_scalar('Train/Precision', train_metrics['avg_precision'], epoch)
        self.writer.add_scalar('Train/Recall', train_metrics['avg_recall'], epoch)
        self.writer.add_scalar('Train/F1', train_metrics['avg_f1'], epoch)
        self.writer.add_scalar('Train/PositiveRate', train_metrics['positive_rate'], epoch)
        self.writer.add_scalar('Train/PredictionRate', train_metrics['prediction_rate'], epoch)
        
        # 记录micro平均指标
        self.writer.add_scalar('Train/mAP_micro', train_metrics['mAP_micro'], epoch)
        self.writer.add_scalar('Train/Precision_micro', train_metrics['precision_micro'], epoch)
        self.writer.add_scalar('Train/Recall_micro', train_metrics['recall_micro'], epoch)
        self.writer.add_scalar('Train/F1_micro', train_metrics['f1_micro'], epoch)
    
    def print_prediction_samples(self, logits, labels, epoch):
        """打印预测样本（仅主进程）"""
        if not is_main_process():
            return
            
        print(f"\n📋 Epoch {epoch+1} 训练样本预测展示 (前5个):")
        
        # 转换为预测标签
        predictions = torch.sigmoid(logits) > 0.5
        
        for i in range(min(5, logits.size(0))):
            # 获取预测的类别索引
            pred_indices = torch.where(predictions[i])[0].cpu().numpy()
            true_indices = torch.where(labels[i] > 0.5)[0].cpu().numpy()
            
            # 计算匹配情况
            correct_preds = set(pred_indices) & set(true_indices)
            
            print(f"  样本{i+1}: 预测={pred_indices.tolist()}, 真实={true_indices.tolist()}, 匹配={len(correct_preds)}/{len(true_indices)}")
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        val_losses = []
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                query_data, support_data, support_masks, batch_info = batch
                
                # 数据移到设备
                query_data = query_data.to(self.device)
                support_data = support_data.to(self.device)
                support_masks = support_masks.to(self.device)
                query_labels = batch_info['query_labels'].to(self.device)
                
                # 前向传播
                results = self.model(query_data, support_data, support_masks)
                
                # 计算损失
                loss = self.criterion(results['logits'], query_labels.float())
                val_losses.append(loss.item())
                
                # 收集预测和标签
                all_logits.append(results['logits'].cpu())
                all_labels.append(query_labels.cpu())
        
        # 计算验证指标
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        metrics = MultiLabelMetrics.compute_metrics(all_logits, all_labels)
        avg_val_loss = np.mean(val_losses)
        
        # 记录验证指标到TensorBoard（仅主进程）
        if is_main_process():
            self.log_validation_metrics(metrics, avg_val_loss, epoch)
        
        return avg_val_loss, metrics
    
    def log_validation_metrics(self, val_metrics, val_loss, epoch):
        """记录验证指标到TensorBoard（仅主进程）"""
        if not is_main_process():
            return
            
        # 记录验证损失和指标（使用'Val/'前缀）
        self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
        self.writer.add_scalar('Val/mAP', val_metrics['mAP'], epoch)
        self.writer.add_scalar('Val/Precision', val_metrics['avg_precision'], epoch)
        self.writer.add_scalar('Val/Recall', val_metrics['avg_recall'], epoch)
        self.writer.add_scalar('Val/F1', val_metrics['avg_f1'], epoch)
        self.writer.add_scalar('Val/PositiveRate', val_metrics['positive_rate'], epoch)
        self.writer.add_scalar('Val/PredictionRate', val_metrics['prediction_rate'], epoch)
        
        # 记录micro平均指标
        self.writer.add_scalar('Val/mAP_micro', val_metrics['mAP_micro'], epoch)
        self.writer.add_scalar('Val/Precision_micro', val_metrics['precision_micro'], epoch)
        self.writer.add_scalar('Val/Recall_micro', val_metrics['recall_micro'], epoch)
        self.writer.add_scalar('Val/F1_micro', val_metrics['f1_micro'], epoch)
    
    def train(self):
        """完整训练流程（支持分布式训练）"""
        if is_main_process():
            print("\n🚀 开始Base Class训练...")
            print(f"  - 训练模式: {'分布式' if self.is_distributed else '单GPU'}")
            if self.is_distributed:
                print(f"  - GPU数量: {self.world_size}")
            print(f"  - 训练轮数: {self.config['num_epochs']}")
            print(f"  - 批大小: {self.config['batch_size']}")
            print(f"  - 类别不均衡比例: 1:{self.config['positive_ratio']}")
            print(f"  - 验证频率: 每个epoch（增强监控）")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # 训练阶段（新增训练指标评估）
            train_loss, train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 每个epoch都进行验证（恢复细粒度监控）
            val_loss, val_metrics = self.validate_epoch(epoch)
            self.val_metrics.append(val_metrics)
            
            # 打印训练和验证信息（仅主进程）
            if is_main_process():
                print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} [训练+验证结果]:")
                print(f"  📈 Train - Loss:{train_loss:.4f}, mAP:{train_metrics['mAP']:.4f}, P:{train_metrics['avg_precision']:.4f}, R:{train_metrics['avg_recall']:.4f}")
                print(f"  📊 Val   - Loss:{val_loss:.4f}, mAP:{val_metrics['mAP']:.4f}, P:{val_metrics['avg_precision']:.4f}, R:{val_metrics['avg_recall']:.4f}")
                
                # 模型保存（仅主进程）
                is_best = val_metrics['mAP'] > self.best_map
                if is_best:
                    self.best_map = val_metrics['mAP']
                    print(f"  🎉 新最佳mAP: {self.best_map:.4f}")
                
                # 保存checkpoint（仅主进程）
                model_to_save = self.model.module if self.is_distributed else self.model
                self.model_manager.save_checkpoint(
                    model=model_to_save,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=is_best
                )
            
            # 同步所有进程（确保所有GPU完成当前epoch再继续）
            if self.is_distributed:
                dist.barrier()
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
        
        if is_main_process():
            print(f"\n✅ 训练完成！最佳mAP: {self.best_map:.4f}")
            if self.writer:
                self.writer.close()


def run_distributed_training(rank, world_size, config):
    """
    分布式训练的工作函数
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        config: 配置字典
    """
    try:
        # 设置分布式环境
        setup_distributed_training(rank, world_size, config)
        
        # 创建训练器
        trainer = BaseClassTrainer(config, rank=rank, world_size=world_size)
        
        # 设置所有组件
        trainer.setup_data_loaders()
        trainer.setup_model()
        trainer.setup_loss_function()
        trainer.setup_optimizer()
        trainer.setup_logging()
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        print(f"❌ Rank {rank} 训练失败: {e}")
        raise e
    finally:
        # 清理分布式环境
        cleanup_distributed_training()


def get_default_config():
    """获取默认配置"""
    return {
        # 数据配置
        'train_query_json': 'datasets/3tab_exp/base_train/3tab_train.json',
        'train_query_dir': 'datasets/3tab_exp/base_train/query_data',
        'val_query_json': 'datasets/3tab_exp/base_train/3tab_val.json',
        'val_query_dir': 'datasets/3tab_exp/base_train/query_data',
        'support_root_dir': 'datasets/3tab_exp/base_train/support_data',
        
        # 模型配置
        'num_classes': 60,
        'sequence_length': 30000,
        'shots_per_class': 1,
        'support_blocks': 0,
        'dropout': 0.15,
        
        # 分类头配置
        'classification_method': 'binary',  # 'binary' 
        'unified_threshold': 0.4,  # unified方法的阈值
        
        # 训练配置
        'num_epochs': 100,
        'batch_size': 8,
        'val_batch_size': 8,
        'num_workers': 0,
        
        # 优化器配置
        'optimizer': 'adam',
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'grad_clip': 1.0,
        
        # 学习率调度
        'scheduler': 'cosine',
        'step_size': 30,
        'gamma': 0.1,
        'min_lr': 1e-6,
        
        # 损失函数配置（处理类别不均衡）
        'loss_type': 'weighted_bce',  # 'weighted_bce', 'focal', 'bce'
        'positive_ratio': 19.0,  # 57/3 ≈ 19，负正样本比例
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        
        # 分布式训练配置
        'use_distributed': False,      # 是否启用分布式训练
        'gpus': [0],                   # 使用的GPU列表
        'dist_backend': 'nccl',        # 分布式后端
        'master_addr': 'localhost',    # 主节点地址
        'master_port': '12355',        # 主节点端口
        
        # 输出配置
        'output_dir': './experiments',
        'save_interval': 10,
    }


def validate_gpu_config(config):
    """
    验证GPU配置的合法性
    
    Args:
        config: 配置字典
        
    Returns:
        validated_config: 验证后的配置
        is_valid: 是否有效
        error_msg: 错误信息
    """
    try:
        # 检查是否有CUDA支持
        if not torch.cuda.is_available():
            if config.get('use_distributed', False):
                return config, False, "❌ CUDA不可用，无法进行GPU训练"
            else:
                print("⚠️ CUDA不可用，将使用CPU训练")
                config['use_distributed'] = False
                return config, True, ""
        
        # 检查GPU数量
        available_gpus = torch.cuda.device_count()
        print(f"🔍 检测到 {available_gpus} 个GPU")
        
        # 验证请求的GPU是否存在
        requested_gpus = config.get('gpus', [0])
        if not isinstance(requested_gpus, list):
            return config, False, "❌ 'gpus' 配置必须是列表"
        
        for gpu_id in requested_gpus:
            if gpu_id >= available_gpus:
                return config, False, f"❌ GPU {gpu_id} 不存在（只有 {available_gpus} 个GPU）"
        
        # 设置分布式训练模式
        num_gpus = len(requested_gpus)
        if num_gpus > 1:
            config['use_distributed'] = True
            print(f"✅ 将使用 {num_gpus} 个GPU进行分布式训练: {requested_gpus}")
        else:
            config['use_distributed'] = False
            print(f"✅ 将使用单GPU训练: GPU {requested_gpus[0]}")
        
        return config, True, ""
        
    except Exception as e:
        return config, False, f"❌ GPU配置验证失败: {e}"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Base Class Training with Multi-GPU Support')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--loss_type', type=str, default='weighted_bce', 
                        choices=['weighted_bce', 'focal', 'bce'], help='损失函数类型')
    parser.add_argument('--classification_method', type=str, default='binary',
                        choices=['binary'], help='分类方法：binary或unified')
    parser.add_argument('--unified_threshold', type=float, default=0.4, 
                        help='unified方法的分类阈值')
    
    # 分布式训练参数
    parser.add_argument('--use_distributed', action='store_true', 
                        help='是否启用分布式训练')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0],
                        help='使用的GPU列表，例如：--gpus 0 1 2 3')
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help='主节点地址（分布式训练）')
    parser.add_argument('--master_port', type=str, default='12355',
                        help='主节点端口（分布式训练）')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_default_config()
    
    # 命令行参数覆盖
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.loss_type:
        config['loss_type'] = args.loss_type
    if args.classification_method:
        config['classification_method'] = args.classification_method
    if args.unified_threshold:
        config['unified_threshold'] = args.unified_threshold
    
    # 分布式训练参数
    if args.use_distributed:
        config['use_distributed'] = True
    if args.gpus:
        config['gpus'] = args.gpus
    if args.master_addr:
        config['master_addr'] = args.master_addr
    if args.master_port:
        config['master_port'] = args.master_port
    
    # 如果提供了配置文件，加载并合并
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    # 验证GPU配置
    config, is_valid, error_msg = validate_gpu_config(config)
    if not is_valid:
        print(error_msg)
        return
    
    print("🚀 开始训练过程...")
    print(f"📋 最终配置:")
    print(f"  - 分布式训练: {config['use_distributed']}")
    print(f"  - 使用GPU: {config['gpus']}")
    print(f"  - 批大小: {config['batch_size']}")
    print(f"  - 学习率: {config['learning_rate']}")
    print(f"  - 分类方法: {config['classification_method']}")
    
    # 启动训练
    if config['use_distributed']:
        # 分布式训练
        world_size = len(config['gpus'])
        try:
            mp.spawn(
                run_distributed_training,
                args=(world_size, config),
                nprocs=world_size,
                join=True
            )
            print("🎉 分布式训练完成！")
        except Exception as e:
            print(f"❌ 分布式训练失败: {e}")
    else:
        # 单GPU训练
        try:
            # 设置CUDA设备
            if torch.cuda.is_available() and config['gpus']:
                torch.cuda.set_device(config['gpus'][0])
            
            # 创建训练器并开始训练
            trainer = BaseClassTrainer(config)
            
            # 设置所有组件
            trainer.setup_data_loaders()
            trainer.setup_model()
            trainer.setup_loss_function()
            trainer.setup_optimizer()
            trainer.setup_logging()
            
            # 开始训练
            trainer.train()
            print("🎉 单GPU训练完成！")
        except Exception as e:
            print(f"❌ 单GPU训练失败: {e}")


if __name__ == '__main__':
    main() 