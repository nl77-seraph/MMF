"""
增强版训练脚本
使用EnhancedMultiMetaFingerNet进行Base Class训练
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
from models.feature_extractors_enhanced import EnhancedMultiMetaFingerNet  # 使用增强版
from utils.metrics import MultiLabelMetrics
from utils.loss_functions import WeightedBCELoss, FocalLoss
from utils.model_manager import ModelManager


# 复用原train.py的辅助函数
from train import (
    setup_distributed_training,
    cleanup_distributed_training,
    is_main_process,
    validate_gpu_config
)


class EnhancedTrainer:
    """
    增强版训练器
    与BaseClassTrainer基本相同，但使用EnhancedMultiMetaFingerNet
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
        
        if is_main_process():
            print(f"🚀 EnhancedTrainer初始化 (混合方案C)")
            print(f"  - 模式: {'分布式训练' if self.is_distributed else '单GPU训练'}")
            if self.is_distributed:
                print(f"  - Rank: {self.rank}/{self.world_size}")
            print(f"  - 设备: {self.device}")
            print(f"  - 改进: SE + Shot Attention + Cross-Class Attention")
    
    def setup_data_loaders(self):
        """设置数据加载器（与原版相同）"""
        if is_main_process():
            print("\n📦 设置数据加载器...")
        
        # 训练数据加载器
        train_loader_base = MetaTrafficDataLoader(
            query_json_path=self.config['train_query_json'],
            query_files_dir=self.config['train_query_dir'],
            support_root_dir=self.config['support_root_dir'],
            activated_classes=list(range(self.config['num_classes'])),
            target_length=self.config['sequence_length'],
            shots_per_class=self.config['shots_per_class'],
            batch_size=self.config['batch_size'],
            shuffle=not self.is_distributed,
            num_workers=self.config['num_workers'],
            random_sampling=True
        )
        
        # 分布式采样器
        if self.is_distributed:
            from torch.utils.data import DataLoader
            self.train_sampler = DistributedSampler(
                train_loader_base.query_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
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
        
        # 验证数据加载器
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
            random_sampling=True
        )
        
        if is_main_process():
            print(f"  ✅ 训练集: {len(self.train_loader)} batches")
            print(f"  ✅ 验证集: {len(self.val_loader)} batches")
    
    def setup_model(self):
        """设置增强版模型"""
        if is_main_process():
            print("\n🧠 设置增强版网络模型...")
        
        # 使用EnhancedMultiMetaFingerNet
        self.model = EnhancedMultiMetaFingerNet(
            num_classes=self.config['num_classes'],
            dropout=self.config['dropout'],
            support_blocks=self.config['support_blocks'],
            classification_method=self.config['classification_method'],
            unified_threshold=self.config['unified_threshold'],
            use_se_in_df=self.config.get('use_se_in_df', False)  # 可选的DF增强
        ).to(self.device)
        
        # DDP包装
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False
            )
            if is_main_process():
                print(f"  ✅ DDP模型包装完成")
        
        # 计算参数量
        if is_main_process():
            model_for_count = self.model.module if self.is_distributed else self.model
            total_params = sum(p.numel() for p in model_for_count.parameters())
            trainable_params = sum(p.numel() for p in model_for_count.parameters() if p.requires_grad)
            
            print(f"  ✅ 模型参数: {total_params:,} 总量, {trainable_params:,} 可训练")
    
    def setup_loss_function(self):
        """设置损失函数（与原版相同）"""
        if is_main_process():
            print("\n⚖️ 设置损失函数...")
        
        positive_ratio = self.config['positive_ratio']
        pos_weight = torch.tensor([positive_ratio] * self.config['num_classes']).to(self.device)
        self.pos_weight = pos_weight
        
        loss_type = self.config['loss_type']
        
        if loss_type == 'weighted_bce':
            self.criterion = WeightedBCELoss(pos_weight=pos_weight)
        elif loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config['focal_alpha'],
                gamma=self.config['focal_gamma'],
                pos_weight=pos_weight
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        if is_main_process():
            print(f"  ✅ 损失函数: {loss_type}")
    
    def setup_optimizer(self):
        """设置优化器（与原版相同）"""
        if is_main_process():
            print("\n🎯 设置优化器...")
        
        model_params = self.model.module.parameters() if self.is_distributed else self.model.parameters()
        
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
    
    def setup_logging(self):
        """设置日志（仅主进程）"""
        if not is_main_process():
            return
            
        print("\n📊 设置日志系统...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_ddp" if self.is_distributed else "_single"
        exp_name = f"enhanced_training_{timestamp}{mode_suffix}"
        self.exp_dir = os.path.join(self.config['output_dir'], exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        log_dir = os.path.join(self.exp_dir, 'logs')
        self.writer = SummaryWriter(log_dir)
        
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.model_manager = ModelManager(checkpoint_dir)
        
        print(f"  ✅ 实验目录: {self.exp_dir}")
    
    def train_epoch(self, epoch):
        """训练一个epoch（与原版逻辑相同）"""
        self.model.train()
        
        if self.is_distributed and self.train_sampler:
            self.train_sampler.set_epoch(epoch)
        
        train_losses = []
        all_train_logits = []
        all_train_labels = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            query_data, support_data, support_masks, batch_info = batch
            
            query_data = query_data.to(self.device)
            support_data = support_data.to(self.device)
            support_masks = support_masks.to(self.device)
            query_labels = batch_info['query_labels'].to(self.device)
            
            results = self.model(query_data, support_data, support_masks)
            loss = self.criterion(results['logits'], query_labels.float())
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.config.get('grad_clip', 0) > 0:
                model_params = self.model.module.parameters() if self.is_distributed else self.model.parameters()
                torch.nn.utils.clip_grad_norm_(model_params, self.config['grad_clip'])
            
            self.optimizer.step()
            
            train_losses.append(loss.item())
            all_train_logits.append(results['logits'].detach().cpu())
            all_train_labels.append(query_labels.detach().cpu())
            
            if is_main_process() and batch_idx % 10 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], step)
        
        avg_train_loss = np.mean(train_losses)
        all_train_logits = torch.cat(all_train_logits, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        train_metrics = MultiLabelMetrics.compute_metrics(all_train_logits, all_train_labels)
        
        if is_main_process():
            self.writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch)
            self.writer.add_scalar('Train/mAP', train_metrics['mAP'], epoch)
        
        return avg_train_loss, train_metrics
    
    def validate_epoch(self, epoch):
        """验证一个epoch（与原版相同）"""
        self.model.eval()
        val_losses = []
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                query_data, support_data, support_masks, batch_info = batch
                
                query_data = query_data.to(self.device)
                support_data = support_data.to(self.device)
                support_masks = support_masks.to(self.device)
                query_labels = batch_info['query_labels'].to(self.device)
                
                results = self.model(query_data, support_data, support_masks)
                loss = self.criterion(results['logits'], query_labels.float())
                val_losses.append(loss.item())
                
                all_logits.append(results['logits'].cpu())
                all_labels.append(query_labels.cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        metrics = MultiLabelMetrics.compute_metrics(all_logits, all_labels)
        avg_val_loss = np.mean(val_losses)
        
        if is_main_process():
            self.writer.add_scalar('Val/EpochLoss', avg_val_loss, epoch)
            self.writer.add_scalar('Val/mAP', metrics['mAP'], epoch)
        
        return avg_val_loss, metrics
    
    def train(self):
        """完整训练流程"""
        if is_main_process():
            print("\n🚀 开始增强版Base Class训练...")
            print(f"  - 训练轮数: {self.config['num_epochs']}")
            print(f"  - 目标: 从0.9+ mAP提升到0.95+ mAP")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            train_loss, train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            val_loss, val_metrics = self.validate_epoch(epoch)
            self.val_metrics.append(val_metrics)
            
            if is_main_process():
                print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}:")
                print(f"  📈 Train - Loss:{train_loss:.4f}, mAP:{train_metrics['mAP']:.4f}")
                print(f"  📊 Val   - Loss:{val_loss:.4f}, mAP:{val_metrics['mAP']:.4f}")
                
                is_best = val_metrics['mAP'] > self.best_map
                if is_best:
                    self.best_map = val_metrics['mAP']
                    print(f"  🎉 新最佳mAP: {self.best_map:.4f}")
                
                model_to_save = self.model.module if self.is_distributed else self.model
                self.model_manager.save_checkpoint(
                    model=model_to_save,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=is_best
                )
            
            if self.is_distributed:
                dist.barrier()
            
            if self.scheduler:
                self.scheduler.step()
        
        if is_main_process():
            print(f"\n✅ 训练完成！最佳mAP: {self.best_map:.4f}")
            if self.writer:
                self.writer.close()


def run_distributed_training(rank, world_size, config):
    """分布式训练工作函数"""
    try:
        setup_distributed_training(rank, world_size, config)
        
        trainer = EnhancedTrainer(config, rank=rank, world_size=world_size)
        trainer.setup_data_loaders()
        trainer.setup_model()
        trainer.setup_loss_function()
        trainer.setup_optimizer()
        trainer.setup_logging()
        trainer.train()
        
    except Exception as e:
        print(f"❌ Rank {rank} 训练失败: {e}")
        raise e
    finally:
        cleanup_distributed_training()


def get_default_config():
    """获取默认配置（与原版相同，添加增强选项）"""
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
        
        # 增强选项
        'use_se_in_df': False,  # 是否在DF中使用SE Block
        
        # 分类头配置
        'classification_method': 'binary',
        'unified_threshold': 0.4,
        
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
        
        # 损失函数配置
        'loss_type': 'weighted_bce',
        'positive_ratio': 19.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        
        # 分布式训练配置
        'use_distributed': False,
        'gpus': [0],
        'dist_backend': 'nccl',
        'master_addr': 'localhost',
        'master_port': '12355',
        
        # 输出配置
        'output_dir': './experiments',
        'save_interval': 10,
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Enhanced Training with Mixed Scheme C')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--use_se_in_df', action='store_true', help='在DF中使用SE Block')
    
    # 分布式训练参数
    parser.add_argument('--use_distributed', action='store_true', help='启用分布式训练')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='GPU列表')
    
    args = parser.parse_args()
    
    config = get_default_config()
    
    # 命令行参数覆盖
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.use_se_in_df:
        config['use_se_in_df'] = True
    if args.use_distributed:
        config['use_distributed'] = True
    if args.gpus:
        config['gpus'] = args.gpus
    
    # 加载配置文件
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    # 验证GPU配置
    config, is_valid, error_msg = validate_gpu_config(config)
    if not is_valid:
        print(error_msg)
        return
    
    print("🚀 开始增强版训练...")
    print(f"📋 配置:")
    print(f"  - 分布式: {config['use_distributed']}")
    print(f"  - GPU: {config['gpus']}")
    print(f"  - DF使用SE: {config['use_se_in_df']}")
    
    # 启动训练
    if config['use_distributed']:
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
            print(f"❌ 训练失败: {e}")
    else:
        try:
            if torch.cuda.is_available() and config['gpus']:
                torch.cuda.set_device(config['gpus'][0])
            
            trainer = EnhancedTrainer(config)
            trainer.setup_data_loaders()
            trainer.setup_model()
            trainer.setup_loss_function()
            trainer.setup_optimizer()
            trainer.setup_logging()
            trainer.train()
            print("🎉 训练完成！")
        except Exception as e:
            print(f"❌ 训练失败: {e}")


if __name__ == '__main__':
    main()


