

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
import argparse
from tqdm import tqdm

def log(msg):
    print(f"[{datetime.now()}][rank {dist.get_rank() if dist.is_initialized() else 0}] {msg}", flush=True)
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.meta_traffic_dataset import QueryTrafficDataset, SupportTrafficDataset
from models.feature_extractors import EnhancedMultiMetaFingerNet
from utils.metrics import MultiLabelMetrics
from utils.metrics import sigmoid
from utils.loss_functions import WeightedBCELoss, FocalLoss, AsymmetricLoss
from utils.model_manager import ModelManager
from utils.misc import setup_distributed_training, cleanup_distributed_training, is_main_process, setup_seed

# GPU配置：根据实际机器修改，或通过环境变量 CUDA_VISIBLE_DEVICES 在命令行指定
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
class RepeatDataset(Dataset):
    """
    重复数据集包装器
    参考 Fewshot_Detection: return loadlines(dataopt['meta']) * cfg.repeat
    
    用于将K-shot的少量数据重复多次以增加训练迭代次数
    """
    
    def __init__(self, base_dataset: Dataset, repeat: int = 1):
        """
        Args:
            base_dataset: 原始数据集
            repeat: 重复次数
        """
        self.base_dataset = base_dataset
        self.repeat = repeat
        self.base_length = len(base_dataset)
    
    def __len__(self):
        return self.base_length * self.repeat
    
    def __getitem__(self, idx):
        # 循环索引到原始数据集
        real_idx = idx % self.base_length
        return self.base_dataset[real_idx]


class FewshotDataLoader:
    """
    Few-shot数据加载器
    复用meta_traffic_dataset的组件，添加repeat支持
    """
    
    def __init__(
        self,
        query_json_path: str,
        query_files_dir: str,
        support_root_dir: str,
        activated_classes: list,
        query_target_length: int = 20000,
        support_target_length: int = 10000,
        shots_per_class: int = 5,
        batch_size: int = 32,
        repeat: int = 1,
        shuffle: bool = True,
        num_workers: int = 4
    ):
        """
        Args:
            query_json_path: Query集索引JSON
            query_files_dir: Query集数据目录
            support_root_dir: Support集根目录
            activated_classes: 激活的类别列表 (base + novel)
            query_target_length: Query序列长度
            support_target_length: Support序列长度
            shots_per_class: 每类support样本数
            batch_size: 批大小
            repeat: 数据重复次数 (参考metatune.data的repeat参数)
            shuffle: 是否打乱
            num_workers: 工作进程数
        """
        self.activated_classes = activated_classes
        self.repeat = repeat
        self.batch_size = batch_size
        
        if is_main_process():
            print(f"\nFewshotDataLoader:")
            print(f"   - {len(activated_classes)}")
            print(f"  - shots_per_class: {shots_per_class}")
            print(f"  - repeat: {repeat}")
            print(f"  - batch_size: {batch_size}")
        
        # 创建Query数据集
        self.query_dataset = QueryTrafficDataset(
            json_index_path=query_json_path,
            query_files_dir=query_files_dir,
            target_length=query_target_length,
            activated_classes=activated_classes
        )
        # 应用repeat包装
        if repeat > 1:
            self.query_dataset_repeated = RepeatDataset(self.query_dataset, repeat)
        else:
            self.query_dataset_repeated = self.query_dataset
        
        # 创建Query DataLoader
        self.query_loader = DataLoader(
            self.query_dataset_repeated,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._query_collate_fn,
            pin_memory=True
        )
        
        # 创建Support数据集 (固定采样模式)
        self.support_dataset = SupportTrafficDataset(
            support_root_dir=support_root_dir,
            activated_classes=activated_classes,
            target_length=support_target_length,
            shots_per_class=shots_per_class,
            random_sampling=True  
        )
        
        # 预加载support数据
        self.support_data, self.support_masks, self.class_order = \
            self.support_dataset.get_all_support_data()
        
        if is_main_process():
            print(f"  - Query(): {len(self.query_dataset)}")
            print(f"  - Query(): {len(self.query_dataset_repeated)}")
            print(f"  - Support: {self.support_data.shape}")
            print(f"   - {len(self.query_loader)}")

    def _query_collate_fn(self, batch):
        """Query集collate函数"""
        query_data_list = []
        query_labels_list = []
        metadata_list = []
        
        for query_data, query_labels, metadata in batch:
            query_data_list.append(query_data)
            query_labels_list.append(query_labels)
            metadata_list.append(metadata)
        
        batch_query_data = torch.stack(query_data_list)
        batch_query_labels = torch.stack(query_labels_list)
        
        return batch_query_data, batch_query_labels, metadata_list
    
    def get_support_data(self):
        """获取support数据"""
        return self.support_data, self.support_masks
    
    def __iter__(self):
        """返回迭代器"""
        return FewshotIterator(self)
    
    def __len__(self):
        return len(self.query_loader)


class FewshotIterator:
    """Few-shot数据迭代器"""
    
    def __init__(self, dataloader: FewshotDataLoader):
        self.dataloader = dataloader
        self.query_iter = iter(dataloader.query_loader)
        self.support_data = dataloader.support_data
        self.support_masks = dataloader.support_masks
    
    def __iter__(self):
        return self
    
    def __next__(self):
        query_data, query_labels, metadata = next(self.query_iter)
        
        batch_info = {
            'query_labels': query_labels,
            'metadata': metadata,
            'class_order': self.dataloader.class_order,
            'num_classes': len(self.dataloader.activated_classes)
        }
        
        return query_data, self.support_data, self.support_masks, batch_info


class FewshotTrainer:
    """
    Few-shot微调训练器
    
    参考Fewshot_Detection的metatune.data配置:
    - neg=0: 学习率因子=1.5
    - repeat: 数据重复
    - dynamic=0: 固定support采样
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
        
        # 混合精度
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
        
        # 日志
        self.writer = None
        self.model_manager = None
        
        if is_main_process():
            print(f"FewshotTrainer init")
            print(f"   - {config.get('finetune_mode', 'full')}")
            print(f"  - K-shot: {config.get('k_shot', 5)}")
            print(f"  - Repeat: {config.get('repeat', 1)}")
            print(f"   - {self.device}")
    
    def setup_data_loaders(self):
        """设置Few-shot数据加载器"""
        if is_main_process():
            print("\nFew-shot...")
        
        # 获取所有类别 (base + novel)
        base_classes = self.config.get('base_classes', list(range(60)))
        novel_classes = self.config.get('novel_classes', [])
        all_classes = sorted(base_classes + novel_classes)
        
        if is_main_process():
            print(f"  - Base classes: {len(base_classes)}")
            print(f"  - Novel classes: {len(novel_classes)} {novel_classes}")
            print(f"   - {len(all_classes)}")
        
        # 训练数据加载器
        self.train_loader = FewshotDataLoader(
            query_json_path=self.config['train_query_json'],
            query_files_dir=self.config['train_query_dir'],
            support_root_dir=self.config['train_support_dir'],
            activated_classes=all_classes,
            query_target_length=self.config['query_target_length'],
            support_target_length=self.config['support_target_length'],
            shots_per_class=self.config['k_shot'],
            batch_size=self.config['batch_size'],
            repeat=self.config.get('repeat', 1),
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        # 验证数据加载器 (不使用repeat)
        if self.config.get('val_query_json'):
            self.val_loader = FewshotDataLoader(
                query_json_path=self.config['val_query_json'],
                query_files_dir=self.config['val_query_dir'],
                support_root_dir=self.config['val_support_dir'],
                activated_classes=all_classes,
                query_target_length=self.config['query_target_length'],
                support_target_length=self.config['support_target_length'],
                shots_per_class=self.config['k_shot'],
                batch_size=self.config['val_batch_size'],
                repeat=1,  # 验证不重复
                shuffle=False,
                num_workers=self.config['num_workers']
            )
        
        if is_main_process():
            print(f"   : {len(self.train_loader)}")
            if self.val_loader:
                print(f"   : {len(self.val_loader)}")
    
    def setup_model(self):
        """设置模型并加载checkpoint"""
        if is_main_process():
            print("\n ...")
        
        # 获取类别数
        base_classes = self.config.get('base_classes', list(range(60)))
        novel_classes = self.config.get('novel_classes', [])
        num_classes = len(base_classes) + len(novel_classes)
        
        # 创建模型

        self.model = EnhancedMultiMetaFingerNet(
            num_classes=num_classes,
            dropout=self.config.get('dropout', 0.15),
            support_blocks=self.config.get('support_blocks', 0),
            use_se_in_df=self.config.get('use_se_in_df', False)
        ).to(self.device)
        
        # 加载checkpoint
        checkpoint_path = self.config.get('checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            if is_main_process():
                print(f"   checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)
            
            # 处理state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 移除module.前缀（如果有）
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # 处理类别数不匹配的情况
            # 如果novel classes增加了类别数，需要调整分类头
            model_state = self.model.state_dict()
            loaded_keys = set(new_state_dict.keys())
            model_keys = set(model_state.keys())
            
            # 找出维度不匹配的层
            mismatched_keys = []
            for key in loaded_keys & model_keys:
                if new_state_dict[key].shape != model_state[key].shape:
                    mismatched_keys.append(key)
                    if is_main_process():
                        print(f"   : {key}")
                        print(f"      checkpoint: {new_state_dict[key].shape}")
                        print(f"      model: {model_state[key].shape}")
            
            # 过滤掉不匹配的键
            filtered_state_dict = {
                k: v for k, v in new_state_dict.items() 
                if k not in mismatched_keys
            }
            
            # 加载权重
            self.model.load_state_dict(filtered_state_dict, strict=False)
            
            if is_main_process():
                print(f"Checkpoint")
                print(f"      {len(filtered_state_dict)}/{len(new_state_dict)}")
                if mismatched_keys:
                    print(f"      {len(mismatched_keys)}")
        else:
            if is_main_process():
                print(f"   checkpoint")
        
        # 应用冻结策略
        self._apply_freeze_strategy()
        
        # DDP包装
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True  # 冻结时可能有未使用参数
            )
            if is_main_process():
                print(f"DDP")
    
    def _apply_freeze_strategy(self):
        """
        应用冻结策略
        
        finetune_mode:
        - head_only: 仅训练classification_head
        - head_meta: 训练classification_head + meta_learnet
        - full: 全模型训练
        """
        finetune_mode = self.config.get('finetune_mode', 'full')
        
        if finetune_mode == 'head_only':
            # 冻结除classification_head外的所有层
            for name, param in self.model.named_parameters():
                if 'classification_head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        elif finetune_mode == 'head_meta':
            # 冻结feature_extractor和feature_reweighting
            for name, param in self.model.named_parameters():
                if 'classification_head' in name or 'meta_learnet' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        else:  # full
            # 全部可训练
            for param in self.model.parameters():
                param.requires_grad = True
        
        # 统计可训练参数
        if is_main_process():
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\n : {finetune_mode}")
            print(f"   : {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
            
            # 显示各模块状态
            module_status = {}
            for name, param in self.model.named_parameters():
                module = name.split('.')[0]
                if module not in module_status:
                    module_status[module] = {'trainable': 0, 'frozen': 0}
                if param.requires_grad:
                    module_status[module]['trainable'] += param.numel()
                else:
                    module_status[module]['frozen'] += param.numel()
            
            print("   :")
            for module, status in module_status.items():
                total = status['trainable'] + status['frozen']
                if status['trainable'] > 0:
                    print(f"     {module}:   ({status['trainable']:,})")
                else:
                    print(f"     {module}:   ({status['frozen']:,})")
    
    def setup_loss_function(self):
        """设置损失函数"""
        if is_main_process():
            print("\n ...")
        
        num_classes = len(self.config.get('base_classes', [])) + len(self.config.get('novel_classes', []))
        positive_ratio = self.config.get('positive_ratio', 10.0)
        pos_weight = torch.tensor([positive_ratio] * num_classes).to(self.device)
        
        loss_type = self.config.get('loss_type', 'weighted_bce')
        
        if loss_type == 'weighted_bce':
            self.criterion = WeightedBCELoss(pos_weight=pos_weight)
        elif loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config.get('focal_alpha', 0.25),
                gamma=self.config.get('focal_gamma', 2.0),
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
            print(f"   : {loss_type}")
    
    def setup_optimizer(self):
        """
        设置优化器
        
        参考Fewshot_Detection的学习率调整:
        - neg_ratio=0 -> factor=1.5
        - learning_rate /= factor
        """
        if is_main_process():
            print("\n ...")
        
        # 获取可训练参数
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        # 学习率调整（参考train_meta.py）
        base_lr = self.config.get('learning_rate', 1e-4)
        neg_ratio = self.config.get('neg_ratio', 0)
        
        # neg_ratio决定学习率因子
        if neg_ratio == 0:
            factor = 1.5
        elif neg_ratio == 1:
            factor = 3.0
        else:
            factor = 1.0
        
        adjusted_lr = base_lr / factor
        
        if is_main_process():
            print(f"  - Base LR: {base_lr}")
            print(f"  - neg_ratio: {neg_ratio} -> factor: {factor}")
            print(f"  - Adjusted LR: {adjusted_lr}")
        
        # 创建优化器
        optimizer_type = self.config.get('optimizer', 'adam')
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=adjusted_lr,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=adjusted_lr,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=adjusted_lr,
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        
        # 调度器
        # 参考Fewshot_Detection: max_epochs = ceil(max_epoch / repeat)
        max_epoch = self.config.get('max_epoch', 2000)
        repeat = self.config.get('repeat', 1)
        effective_epochs = int(np.ceil(max_epoch / 50))
        
        if is_main_process():
            print(f"  - max_epoch: {max_epoch}, repeat: {repeat}")
            print(f"  - effective_epochs: {effective_epochs}")
        
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=effective_epochs,
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 10),
                gamma=self.config.get('gamma', 0.1)
            )
        
        if is_main_process():
            print(f"   : {optimizer_type}")
            print(f"   : {scheduler_type}")
        
        # 保存effective_epochs供训练使用
        self.effective_epochs = effective_epochs
    
    def setup_logging(self):
        """设置日志"""
        if not is_main_process():
            return
        
        print("\n ...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        k_shot = self.config.get('k_shot', 5)
        finetune_mode = self.config.get('finetune_mode', 'full')
        exp_name = f"fewshot_{k_shot}shot_{finetune_mode}_{timestamp}"
        
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
        
        print(f"   : {self.exp_dir}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        train_losses = []
        batch_times = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()
            
            query_data, support_data, support_masks, batch_info = batch
            
            query_data = query_data.to(self.device, non_blocking=True)
            support_data = support_data.to(self.device, non_blocking=True)
            support_masks = support_masks.to(self.device, non_blocking=True)
            query_labels = batch_info['query_labels'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with autocast():
                    results = self.model(query_data, support_data, support_masks)
                    loss = self.criterion(results['logits'], query_labels.float())
                
                self.scaler.scale(loss).backward()
                
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    params = self.model.module.parameters() if self.is_distributed else self.model.parameters()
                    torch.nn.utils.clip_grad_norm_(params, self.config['grad_clip'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                results = self.model(query_data, support_data, support_masks)
                loss = self.criterion(results['logits'], query_labels.float())
                
                loss.backward()
                
                if self.config.get('grad_clip', 0) > 0:
                    params = self.model.module.parameters() if self.is_distributed else self.model.parameters()
                    torch.nn.utils.clip_grad_norm_(params, self.config['grad_clip'])
                
                self.optimizer.step()
            
            train_losses.append(loss.item())
            batch_times.append(time.time() - batch_start)
            
            if is_main_process() and batch_idx % 20 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], step)
        
        avg_loss = np.mean(train_losses)
        avg_time = np.mean(batch_times)
        
        if is_main_process():
            self.writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        
        return avg_loss, avg_time
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        if self.val_loader is None:
            return 0.0, {}
        
        self.model.eval()
        val_losses = []
        all_logits = []
        all_labels = []
        
        # 收集前5个batch的预测和标签用于详细输出
        first_5_batches_logits = []
        first_5_batches_labels = []
        first_5_batches_metadata = []
        batch_count = 0
        
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
                batch_logits = results['logits'].float().cpu()
                batch_labels = query_labels.cpu()
                
                all_logits.append(batch_logits)
                all_labels.append(batch_labels)
                
                # 收集前5个batch
                if batch_count < 5:
                    first_5_batches_logits.append(batch_logits)
                    first_5_batches_labels.append(batch_labels)
                    first_5_batches_metadata.append(batch_info['metadata'])
                    batch_count += 1
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        metrics = MultiLabelMetrics.compute_metrics(all_logits, all_labels, self.config)
        avg_loss = np.mean(val_losses)
        
        # 计算novel classes的详细指标
        base_classes = self.config.get('base_classes', list(range(60)))
        novel_classes = self.config.get('novel_classes', [])
        all_classes = sorted(base_classes + novel_classes)
        
        novel_metrics = MultiLabelMetrics.compute_novel_class_metrics(
            all_logits, 
            all_labels, 
            novel_classes=novel_classes,
            activated_classes=all_classes,
            threshold=0.5,
            k=self.config.get('tabs','3')
        )
        metrics['novel_metrics'] = novel_metrics
        
        if is_main_process():
            self.writer.add_scalar('Val/EpochLoss', avg_loss, epoch)
            self.writer.add_scalar('Val/sig_mAP', metrics['sig_mAP'], epoch)
            self.writer.add_scalar('Val/pk', metrics['pk'], epoch)
            self.writer.add_scalar('Val/Novel_Avg_Precision', novel_metrics['avg_precision'], epoch)
            self.writer.add_scalar('Val/Novel_Avg_Recall', novel_metrics['avg_recall'], epoch)
            self.writer.add_scalar('Val/Novel_Avg_F1', novel_metrics['avg_f1'], epoch)
            
            # 打印novel classes详细指标
            MultiLabelMetrics.print_novel_class_metrics(novel_metrics, novel_classes)
            
            # 打印前5个batch的预测结果
            # self._print_first_5_batches(
            #     first_5_batches_logits, 
            #     first_5_batches_labels, 
            #     first_5_batches_metadata,
            #     all_classes,
            #     novel_classes
            # )
        
        return avg_loss, metrics
    
    def _print_first_5_batches(self, batches_logits, batches_labels, batches_metadata, 
                               activated_classes, novel_classes):
        """
        打印前5个batch的预测结果和真实标签
        
        Args:
            batches_logits: 前5个batch的logits列表
            batches_labels: 前5个batch的标签列表
            batches_metadata: 前5个batch的元数据列表
            activated_classes: 所有激活的类别列表
            novel_classes: novel类别列表
        """
        print("\n" + "="*80)
        print(" 5Batch")
        print("="*80)
        
        for batch_idx, (batch_logits, batch_labels, batch_metadata) in enumerate(
            zip(batches_logits, batches_labels, batches_metadata)
        ):
            print(f"\n--- Batch {batch_idx + 1} ---")
            
            # 转换为numpy
            logits_np = batch_logits.numpy()
            labels_np = batch_labels.numpy()
            probs_np = sigmoid(logits_np)
            
            batch_size = logits_np.shape[0]
            
            for sample_idx in range(min(3, batch_size)):  # 每个batch最多显示3个样本
                print(f"\n   {sample_idx + 1}:")
                
                # 获取真实标签
                true_label_indices = np.where(labels_np[sample_idx] > 0.5)[0]
                true_labels = [activated_classes[idx] for idx in true_label_indices]
                
                # 获取预测标签（top-k，k=真实标签数）
                k = len(true_label_indices) if len(true_label_indices) > 0 else 1
                top_k_indices = np.argsort(probs_np[sample_idx])[-k:][::-1]
                pred_labels = [activated_classes[idx] for idx in top_k_indices]
                pred_probs = [probs_np[sample_idx][idx] for idx in top_k_indices]
                
                # 分离base和novel
                true_base = [l for l in true_labels if l not in novel_classes]
                true_novel = [l for l in true_labels if l in novel_classes]
                pred_base = [l for l in pred_labels if l not in novel_classes]
                pred_novel = [l for l in pred_labels if l in novel_classes]
                
                print(f"     (Base): {true_base}")
                print(f"     (Novel): {true_novel}")
                print(f"     (Base): {pred_base}")
                print(f"     (Novel): {pred_novel}")
                
                # 显示novel classes的预测概率
                if novel_classes:
                    novel_probs = []
                    class_to_idx = {cls_id: idx for idx, cls_id in enumerate(activated_classes)}
                    for novel_cls in novel_classes:
                        if novel_cls in class_to_idx:
                            idx = class_to_idx[novel_cls]
                            prob = probs_np[sample_idx][idx]
                            novel_probs.append((novel_cls, prob))
                    
                    if novel_probs:
                        novel_probs_str = ", ".join([f"C{cls}:{prob:.3f}" for cls, prob in novel_probs])
                        print(f"Novel: {novel_probs_str}")
                
                # 计算该样本的匹配情况
                correct_base = len(set(true_base) & set(pred_base))
                correct_novel = len(set(true_novel) & set(pred_novel))
                print(f"    : Base={correct_base}/{len(true_base)}, Novel={correct_novel}/{len(true_novel)}")
                
                # 显示文件名（如果有）
                if sample_idx < len(batch_metadata):
                    metadata = batch_metadata[sample_idx]
                    if 'filename' in metadata:
                        print(f"    : {metadata['filename']}")
        
        print("\n" + "="*80)
    
    def train(self):
        """完整训练流程"""
        if is_main_process():
            print("\n" + "="*60)
            print(" Few-shot")
            print("="*60)
            print(f"  - K-shot: {self.config.get('k_shot', 5)}")
            print(f"  - Repeat: {self.config.get('repeat', 1)}")
            print(f"   - {self.config.get('finetune_mode', 'full')}")
            print(f"  - Effective epochs: {self.effective_epochs}")

        # 初始评估（未微调）
        if is_main_process():
            print("\n ...")
        init_val_loss, init_val_metrics = self.validate_epoch(-1)
        if is_main_process() and init_val_metrics:
            print(f"Init Val Loss: {init_val_loss:.4f}")
            MultiLabelMetrics.print_metrics_summary(init_val_metrics)
            self.best_map = max(self.best_map, init_val_metrics.get('sig_mAP', 0.0))
        if self.is_distributed:
            dist.barrier()
        
        for epoch in range(self.effective_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            #log("train_epoch start")
            train_loss, avg_batch_time = self.train_epoch(epoch)
            #log("train_epoch done")

            #log("validate_epoch start")
            val_loss, val_metrics = self.validate_epoch(epoch)
            #log("validate_epoch done")
            
            epoch_time = time.time() - epoch_start
            
            if is_main_process():
                print(f"\nEpoch {epoch+1}/{self.effective_epochs} | Time: {epoch_time:.1f}s")
                print(f"Train Loss: {train_loss:.4f}")
                if val_metrics:
                    print(f"Val Loss: {val_loss:.4f}")
                    MultiLabelMetrics.print_metrics_summary(val_metrics)
                    
                    is_best = val_metrics.get('sig_mAP', 0) > self.best_map
                    if is_best:
                        self.best_map = val_metrics['sig_mAP']
                        print(f"    sig_mAP: {self.best_map:.4f}")
                    
                    model_to_save = self.model.module if self.is_distributed else self.model
                    t0 = time.time()
                    #log(f"save_checkpoint start, time: {t0}")
                    self.model_manager.save_checkpoint(
                        model=model_to_save,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=val_metrics,
                        is_best=is_best
                    )
                    #log(f"save_checkpoint done, time: {time.time() - t0}")
            if self.is_distributed:
                #log("barrier start")
                dist.barrier()
                #log("barrier done")
            if self.scheduler:
                self.scheduler.step()
        
        if is_main_process():
            print(f"\nFew-shotmAP: {self.best_map:.4f}")
            if self.writer:
                self.writer.close()


def run_distributed_training(rank, world_size, config):
    """分布式训练入口"""
    try:
        setup_distributed_training(rank, world_size, config)
        
        trainer = FewshotTrainer(config, rank=rank, world_size=world_size)
        trainer.setup_data_loaders()
        trainer.setup_model()
        trainer.setup_loss_function()
        trainer.setup_optimizer()
        trainer.setup_logging()
        trainer.train()
        
    except Exception as e:
        print(f" Rank {rank} : {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        cleanup_distributed_training()


def get_fewshot_config():
    """获取Few-shot配置"""
    parser = argparse.ArgumentParser(description='Few-shot Fine-tuning')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f" : {args.config}")
        return None
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 验证必要字段
    required_fields = [
        'train_query_json', 'train_query_dir', 'train_support_dir',
        'base_classes', 'novel_classes', 'k_shot'
    ]
    
    for field in required_fields:
        if field not in config:
            print(f" : {field}")
            return None
    
    # GPU验证
    if torch.cuda.is_available():
        gpus = config.get('gpus', [0])
        available_gpus = torch.cuda.device_count()
        
        for gpu in gpus:
            if gpu >= available_gpus:
                print(f" GPU {gpu}")
                return None
        
        config['use_distributed'] = len(gpus) > 1
        print(f" GPU: {gpus}")
    else:
        config['use_distributed'] = False
        print(" CUDACPU")
    
    return config


def main():
    config = get_fewshot_config()
    if config is None:
        return
    
    setup_seed(config.get('seed', 42))
    
    if config['use_distributed']:
        world_size = len(config['gpus'])
        try:
            mp.spawn(
                run_distributed_training,
                args=(world_size, config),
                nprocs=world_size,
                join=True
            )
            print("")
        except Exception as e:
            print(f" : {e}")
    else:
        try:
            if torch.cuda.is_available() and config.get('gpus'):
                torch.cuda.set_device(config['gpus'][0])
            
            trainer = FewshotTrainer(config)
            trainer.setup_data_loaders()
            trainer.setup_model()
            trainer.setup_loss_function()
            trainer.setup_optimizer()
            trainer.setup_logging()
            trainer.train()
            print("")
        except Exception as e:
            print(f" : {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

