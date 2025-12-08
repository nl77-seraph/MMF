"""
优化版训练脚本
使用OptimizedMultiMetaFingerNet进行Base Class训练

主要优化:
1. 轻量级MetaLearnet
2. 并行化Classification Head
3. 可选关闭Cross-Attention
4. 支持混合精度训练加速
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.meta_traffic_dataloader import MetaTrafficDataLoader
from models.feature_extractors_optimized import OptimizedMultiMetaFingerNet
from utils.metrics import MultiLabelMetrics
from utils.loss_functions import WeightedBCELoss, FocalLoss, AsymmetricLoss
from utils.model_manager import ModelManager
from utils.misc import *


class OptimizedTrainer:
    """
    优化版训练器
    
    优化点:
    1. 使用OptimizedMultiMetaFingerNet
    2. 支持混合精度训练 (AMP)
    3. 训练过程性能分析
    """
    
    def __init__(self, config, rank=None, world_size=None):
        self.config = config
        self.rank = rank if rank is not None else 0
        self.world_size = world_size if world_size is not None else 1
        self.is_distributed = world_size is not None and world_size > 1
        
        if self.is_distributed:
            self.device = torch.device(f'cuda:{rank}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 训练组件
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
        
        # 日志
        self.writer = None
        self.model_manager = None
        
        # 性能分析
        self.timing_stats = {
            'data_load': [],
            'forward': [],
            'loss': [],
            'backward': [],
            'step': [],
            'total': []
        }
        
        if is_main_process():
            print(f"🚀 OptimizedTrainer 初始化")
            print(f"  - 模式: {'分布式' if self.is_distributed else '单GPU'}")
            print(f"  - 设备: {self.device}")
            print(f"  - 混合精度: {'开启' if self.use_amp else '关闭'}")
    
    def setup_data_loaders(self):
        """设置数据加载器"""
        if is_main_process():
            print("\n📦 设置数据加载器...")
        
        train_loader_base = MetaTrafficDataLoader(
            query_json_path=self.config['train_query_json'],
            query_files_dir=self.config['train_query_dir'],
            support_root_dir=self.config['train_support_root_dir'],
            activated_classes=list(range(self.config['num_classes'])),
            query_target_length=self.config['query_target_length'],
            support_target_length=self.config['support_target_length'],
            shots_per_class=self.config['shots_per_class'],
            batch_size=self.config['batch_size'],
            shuffle=not self.is_distributed,
            num_workers=self.config['num_workers'],
            random_sampling=True
        )
        
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
                collate_fn=train_loader_base._query_collate_fn,
                pin_memory=True  # 加速数据传输
            )
        else:
            self.train_sampler = None
        
        self.train_loader = train_loader_base
        
        self.val_loader = MetaTrafficDataLoader(
            query_json_path=self.config['val_query_json'],
            query_files_dir=self.config['val_query_dir'],
            support_root_dir=self.config['val_support_root_dir'],
            activated_classes=list(range(self.config['num_classes'])),
            query_target_length=self.config['query_target_length'],
            support_target_length=self.config['support_target_length'],
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
        """设置优化版模型"""
        if is_main_process():
            print("\n🧠 设置OptimizedMultiMetaFingerNet...")
        
        self.model = OptimizedMultiMetaFingerNet(
            num_classes=self.config['num_classes'],
            dropout=self.config['dropout'],
            use_cross_attention=self.config.get('use_cross_attention', False),
            num_topm_layers=self.config.get('num_topm_layers', 2),
            meta_learnet_type=self.config.get('meta_learnet_type', 'lightweight')
        ).to(self.device)
        
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False  # 优化：关闭未使用参数检查
            )
            if is_main_process():
                print(f"  ✅ DDP包装完成")
    
    def setup_loss_function(self):
        """设置损失函数"""
        if is_main_process():
            print("\n⚖️ 设置损失函数...")
        
        positive_ratio = self.config['positive_ratio']
        pos_weight = torch.tensor([positive_ratio] * self.config['num_classes']).to(self.device)
        
        loss_type = self.config['loss_type']
        
        if loss_type == 'weighted_bce':
            self.criterion = WeightedBCELoss(pos_weight=pos_weight)
        elif loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config['focal_alpha'],
                gamma=self.config['focal_gamma'],
                pos_weight=pos_weight
            )
        elif loss_type == 'asy':
            self.criterion = AsymmetricLoss(
                gamma_pos=self.config.get('asy_gamma_pos', 0.0),
                gamma_neg=self.config.get('asy_gamma_neg', 4.0),
                clip=self.config.get('asy_clip', 0.05)
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        if is_main_process():
            print(f"  ✅ 损失函数: {loss_type}")
            print(f"  ✅ positive_ratio: {positive_ratio}")
    
    def setup_optimizer(self):
        """设置优化器"""
        if is_main_process():
            print("\n🎯 设置优化器...")
        
        model_params = self.model.module.parameters() if self.is_distributed else self.model.parameters()
        
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                model_params,
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
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
        elif self.config['scheduler'] == 'onecycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'],
                epochs=self.config['num_epochs'],
                steps_per_epoch=len(self.train_loader)
            )
        
        if is_main_process():
            print(f"  ✅ 优化器: {self.config['optimizer']}")
            print(f"  ✅ 学习率: {self.config['learning_rate']}")
            print(f"  ✅ 调度器: {self.config['scheduler']}")
    
    def setup_logging(self):
        """设置日志"""
        if not is_main_process():
            return
            
        print("\n📊 设置日志系统...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_ddp" if self.is_distributed else "_single"
        exp_name = f"{self.config['model_name']}_{timestamp}{mode_suffix}"
        self.exp_dir = os.path.join(self.config['output_dir'], exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        log_dir = os.path.join(self.exp_dir, 'logs')
        self.writer = SummaryWriter(log_dir)
        
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.model_manager = ModelManager(checkpoint_dir)
        
        print(f"  ✅ 实验目录: {self.exp_dir}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        if self.is_distributed and self.train_sampler:
            self.train_sampler.set_epoch(epoch)
        
        train_losses = []
        batch_times = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()
            
            query_data, support_data, support_masks, batch_info = batch
            
            # 数据移动到GPU
            query_data = query_data.to(self.device, non_blocking=True)
            support_data = support_data.to(self.device, non_blocking=True)
            support_masks = support_masks.to(self.device, non_blocking=True)
            query_labels = batch_info['query_labels'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            
            # 混合精度前向传播
            if self.use_amp:
                with autocast():
                    results = self.model(query_data, support_data, support_masks)
                    loss = self.criterion(results['logits'], query_labels.float())
                
                self.scaler.scale(loss).backward()
                
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    model_params = self.model.module.parameters() if self.is_distributed else self.model.parameters()
                    torch.nn.utils.clip_grad_norm_(model_params, self.config['grad_clip'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                results = self.model(query_data, support_data, support_masks)
                loss = self.criterion(results['logits'], query_labels.float())
                
                loss.backward()
                
                if self.config.get('grad_clip', 0) > 0:
                    model_params = self.model.module.parameters() if self.is_distributed else self.model.parameters()
                    torch.nn.utils.clip_grad_norm_(model_params, self.config['grad_clip'])
                
                self.optimizer.step()
            
            # OneCycle scheduler per batch
            if self.config['scheduler'] == 'onecycle':
                self.scheduler.step()
            
            train_losses.append(loss.item())
            batch_times.append(time.time() - batch_start)
            
            # 日志
            if is_main_process() and batch_idx % 20 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], step)
                
                avg_batch_time = np.mean(batch_times[-20:]) if len(batch_times) >= 20 else np.mean(batch_times)
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f} | Time: {avg_batch_time*1000:.1f}ms")
        
        avg_train_loss = np.mean(train_losses)
        avg_batch_time = np.mean(batch_times)
        
        if is_main_process():
            self.writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch)
            self.writer.add_scalar('Train/BatchTime', avg_batch_time * 1000, epoch)
        
        return avg_train_loss, avg_batch_time
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        val_losses = []
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                query_data, support_data, support_masks, batch_info = batch
                
                query_data = query_data.to(self.device, non_blocking=True)
                support_data = support_data.to(self.device, non_blocking=True)
                support_masks = support_masks.to(self.device, non_blocking=True)
                query_labels = batch_info['query_labels'].to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        results = self.model(query_data, support_data, support_masks)
                        loss = self.criterion(results['logits'], query_labels.float())
                else:
                    results = self.model(query_data, support_data, support_masks)
                    loss = self.criterion(results['logits'], query_labels.float())
                
                val_losses.append(loss.item())
                all_logits.append(results['logits'].float().cpu())
                all_labels.append(query_labels.cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = MultiLabelMetrics.compute_metrics(all_logits, all_labels, self.config)
        avg_val_loss = np.mean(val_losses)
        
        if is_main_process():
            self.writer.add_scalar('Val/EpochLoss', avg_val_loss, epoch)
            self.writer.add_scalar('Val/soft_mAP', metrics['soft_mAP'], epoch)
            self.writer.add_scalar('Val/sig_mAP', metrics['sig_mAP'], epoch)
            self.writer.add_scalar('Val/pk', metrics['pk'], epoch)
            self.writer.add_scalar('Val/mapk', metrics['mapk'], epoch)
        
        return avg_val_loss, metrics
    
    def train(self):
        """完整训练流程"""
        if is_main_process():
            print("\n" + "="*60)
            print("🚀 开始优化版Base Class训练")
            print("="*60)
            print(f"  - 训练轮数: {self.config['num_epochs']}")
            print(f"  - MetaLearnet类型: {self.config.get('meta_learnet_type', 'lightweight')}")
            print(f"  - Cross-Attention: {'开启' if self.config.get('use_cross_attention', False) else '关闭'}")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # 训练
            train_loss, avg_batch_time = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_metrics = self.validate_epoch(epoch)
            self.val_metrics.append(val_metrics)
            
            epoch_time = time.time() - epoch_start
            
            if is_main_process():
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{self.config['num_epochs']} | Time: {epoch_time:.1f}s")
                print(f"{'='*60}")
                print(f"  📈 Train Loss: {train_loss:.4f} | Avg Batch: {avg_batch_time*1000:.1f}ms")
                print(f"  📊 Val Loss: {val_loss:.4f}")
                MultiLabelMetrics.print_metrics_summary(val_metrics)
                
                is_best = val_metrics['sig_mAP'] > self.best_map
                if is_best:
                    self.best_map = val_metrics['sig_mAP']
                    print(f"  🎉 新最佳 sig_mAP: {self.best_map:.4f}")
                
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
            
            # 非OneCycle调度器在epoch结束后更新
            if self.scheduler and self.config['scheduler'] != 'onecycle':
                self.scheduler.step()
        
        if is_main_process():
            print(f"\n✅ 训练完成！最佳mAP: {self.best_map:.4f}")
            if self.writer:
                self.writer.close()


def run_distributed_training(rank, world_size, config):
    """分布式训练入口"""
    try:
        setup_distributed_training(rank, world_size, config)
        
        trainer = OptimizedTrainer(config, rank=rank, world_size=world_size)
        trainer.setup_data_loaders()
        trainer.setup_model()
        trainer.setup_loss_function()
        trainer.setup_optimizer()
        trainer.setup_logging()
        trainer.train()
        
    except Exception as e:
        print(f"❌ Rank {rank} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        cleanup_distributed_training()


def main():
    """主函数"""
    config = get_final_config()
    if config is None:
        return
    
    setup_seed(config['seed'])
    
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
            
            trainer = OptimizedTrainer(config)
            trainer.setup_data_loaders()
            trainer.setup_model()
            trainer.setup_loss_function()
            trainer.setup_optimizer()
            trainer.setup_logging()
            trainer.train()
            print("🎉 训练完成！")
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()



