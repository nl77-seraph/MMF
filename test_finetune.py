"""
Few-shot Fine-tuning 测试脚本

功能：
- 加载训练好的模型（basetrain或finetune）
- 仅进行一轮测试（使用val数据集）
- 输出详细的评估指标
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.meta_traffic_dataset import QueryTrafficDataset, SupportTrafficDataset
from models.feature_extractors import EnhancedMultiMetaFingerNet
from utils.metrics import MultiLabelMetrics
from utils.metrics import sigmoid
from utils.misc import setup_seed

# GPU配置：根据实际机器修改，或通过环境变量 CUDA_VISIBLE_DEVICES 在命令行指定
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


class FewshotTestDataLoader:
    """Few-shot测试数据加载器"""
    
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
        num_workers: int = 4
    ):
        self.activated_classes = activated_classes
        self.batch_size = batch_size
        
        print(f"FewshotTestDataLoader init:")
        print(f"  - classes: {len(activated_classes)}")
        print(f"  - shots_per_class: {shots_per_class}")
        print(f"  - batch_size: {batch_size}")
        
        # 创建Query数据集
        self.query_dataset = QueryTrafficDataset(
            json_index_path=query_json_path,
            query_files_dir=query_files_dir,
            target_length=query_target_length,
            activated_classes=activated_classes
        )
        
        # 创建Query DataLoader
        self.query_loader = DataLoader(
            self.query_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._query_collate_fn,
            pin_memory=True
        )
        
        # 创建Support数据集
        self.support_dataset = SupportTrafficDataset(
            support_root_dir=support_root_dir,
            activated_classes=activated_classes,
            target_length=support_target_length,
            shots_per_class=shots_per_class,
            random_sampling=False  # 测试时不随机采样
        )
        
        # 预加载support数据
        self.support_data, self.support_masks, self.class_order = \
            self.support_dataset.get_all_support_data()
        
        print(f"  - query samples: {len(self.query_dataset)}")
        print(f"  - support shape: {self.support_data.shape}")
        print(f"  - batches: {len(self.query_loader)}")

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
        return FewshotTestIterator(self)
    
    def __len__(self):
        return len(self.query_loader)


class FewshotTestIterator:
    """Few-shot测试数据迭代器"""
    
    def __init__(self, dataloader: FewshotTestDataLoader):
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


class FewshotTester:
    """Few-shot模型测试器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 混合精度
        self.use_amp = config.get('use_amp', True)
        
        print(f"FewshotTester init")
        print(f"   - {self.device}")
        print(f"  - K-shot: {config.get('k_shot', 5)}")
        print(f"  - Tab: {config.get('tabs', '3')}")
    
    def setup_data_loader(self):
        """设置测试数据加载器"""
        print("\n ...")
        
        # 获取所有类别 (base + novel)
        base_classes = self.config.get('base_classes', list(range(60)))
        novel_classes = self.config.get('novel_classes', [])
        all_classes = sorted(base_classes + novel_classes)
        
        print(f"  - Base classes: {len(base_classes)}")
        print(f"  - Novel classes: {len(novel_classes)} {novel_classes}")
        print(f"   - {len(all_classes)}")
        
        # 测试数据加载器
        self.test_loader = FewshotTestDataLoader(
            query_json_path=self.config['test_query_json'],
            query_files_dir=self.config['test_query_dir'],
            support_root_dir=self.config['test_support_dir'],
            activated_classes=all_classes,
            query_target_length=self.config['query_target_length'],
            support_target_length=self.config['support_target_length'],
            shots_per_class=self.config['k_shot'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers']
        )
        
        print(f"   : {len(self.test_loader)}")
    
    def setup_model(self):
        """设置模型并加载checkpoint"""
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
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint不存在: {checkpoint_path}")
        
        print(f"   checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
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
        model_state = self.model.state_dict()
        loaded_keys = set(new_state_dict.keys())
        model_keys = set(model_state.keys())
        
        # 找出维度不匹配的层
        mismatched_keys = []
        for key in loaded_keys & model_keys:
            if new_state_dict[key].shape != model_state[key].shape:
                mismatched_keys.append(key)
                print(f"   : {key}")
                print(f"      checkpoint: {new_state_dict[key].shape}")
                print(f"      model: {model_state[key].shape}")
        
        # 过滤掉不匹配的键
        filtered_state_dict = {
            k: v for k, v in new_state_dict.items() 
            if k not in mismatched_keys
        }
        
        # 加载权重
        missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"Checkpoint")
        print(f"      {len(filtered_state_dict)}/{len(new_state_dict)}")
        if mismatched_keys:
            print(f"      {len(mismatched_keys)}")
        if missing_keys:
            print(f"     : {len(missing_keys)}")
        if unexpected_keys:
            print(f"     : {len(unexpected_keys)}")
        
        # 设置为评估模式
        self.model.eval()
    
    def test(self):
        """执行测试"""
        print("\n" + "="*60)
        print(" Few-shot")
        print("="*60)
        
        test_start = time.time()
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing", ncols=100)
            for batch in pbar:
                query_data, support_data, support_masks, batch_info = batch
                
                query_data = query_data.to(self.device, non_blocking=True)
                support_data = support_data.to(self.device, non_blocking=True)
                support_masks = support_masks.to(self.device, non_blocking=True)
                query_labels = batch_info['query_labels'].to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        results = self.model(query_data, support_data, support_masks)
                else:
                    results = self.model(query_data, support_data, support_masks)
                
                batch_logits = results['logits'].float().cpu()
                batch_labels = query_labels.cpu()
                
                all_logits.append(batch_logits)
                all_labels.append(batch_labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算基础指标
        metrics = MultiLabelMetrics.compute_metrics(all_logits, all_labels, self.config)
        
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
            k=self.config.get('tabs', '3')
        )
        metrics['novel_metrics'] = novel_metrics
        
        test_time = time.time() - test_start
        
        # 打印结果
        print("\n" + "="*60)
        print("-" * 60)
        print("="*60)
        print(f": {test_time:.2f}s")
        print(f": {len(all_labels)}")
        print(f": {len(self.test_loader)}")
        print()
        
        MultiLabelMetrics.print_metrics_summary(metrics)
        print()
        MultiLabelMetrics.print_novel_class_metrics(novel_metrics, novel_classes)
        
        # 保存预测数据用于后续PR曲线绘制
        self._save_predictions(all_logits, all_labels, base_classes, novel_classes)
        
        return metrics
    
    
    def _save_results(self, metrics, test_time):
        """保存测试结果"""
        output_dir = self.config.get('output_dir', './test_results')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        k_shot = self.config.get('k_shot', 5)
        tabs = self.config.get('tabs', '3')
        
        result_file = os.path.join(output_dir, f'test_results_{tabs}tab_{k_shot}shot_{timestamp}.json')
        
        # 准备保存的数据
        results = {
            'config': {
                'checkpoint_path': self.config.get('checkpoint_path'),
                'k_shot': k_shot,
                'tabs': tabs,
                'base_classes': self.config.get('base_classes'),
                'novel_classes': self.config.get('novel_classes')
            },
            'test_time': test_time,
            'metrics': {
                'sig_mAP': float(metrics['sig_mAP']),
                'pk': float(metrics['pk']),
                'novel_metrics': {
                    'per_class_precision': {int(k): float(v) for k, v in metrics['novel_metrics']['per_class_precision'].items()},
                    'per_class_recall': {int(k): float(v) for k, v in metrics['novel_metrics']['per_class_recall'].items()},
                    'per_class_f1': {int(k): float(v) for k, v in metrics['novel_metrics']['per_class_f1'].items()},
                    'avg_precision': float(metrics['novel_metrics']['avg_precision']),
                    'avg_recall': float(metrics['novel_metrics']['avg_recall']),
                    'avg_f1': float(metrics['novel_metrics']['avg_f1'])
                }
            },
            'timestamp': timestamp
        }
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n : {result_file}")
    
    
    def _save_predictions(self, logits, targets, base_classes, novel_classes):
        """保存预测数据用于后续分析和PR曲线绘制"""
        output_dir = self.config.get('output_dir', './test_results')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        k_shot = self.config.get('k_shot', 5)
        tabs = self.config.get('tabs', '3')
        
        pred_file = os.path.join(output_dir, f'predictions_{tabs}tab_{k_shot}shot_{timestamp}.npz')
        
        # 转换为numpy
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        os.makedirs(os.path.dirname(pred_file), exist_ok=True)
        # 保存数据
        np.savez_compressed(
            pred_file,
            logits=logits,
            targets=targets,
            base_classes=np.array(base_classes),
            novel_classes=np.array(novel_classes),
            tabs=tabs,
            k_shot=k_shot,
            timestamp=timestamp
        )
        
        print(f" : {pred_file}")
        print(f"   - Logits shape: {logits.shape}")
        print(f"   - Targets shape: {targets.shape}")
        print(f"   - Base classes: {len(base_classes)}")
        print(f"   - Novel classes: {len(novel_classes)}")
    


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise ValueError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Few-shot Fine-tuning Test')
    parser.add_argument('--config', type=str, required=True, help='测试配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print(f" : {args.config}")
    
    # 设置随机种子
    setup_seed(config.get('seed', 42))
    
    # 创建测试器
    tester = FewshotTester(config)
    
    # 设置数据加载器
    tester.setup_data_loader()
    
    # 设置模型
    tester.setup_model()
    
    # 执行测试
    metrics = tester.test()
    
    print("\n")
    print(f"   sig_mAP: {metrics['sig_mAP']:.4f}")
    print(f"   pk: {metrics['pk']:.4f}")
    print(f"Novel Avg Precision: {metrics['novel_metrics']['avg_precision']:.4f}")
    print(f"Novel Avg Recall: {metrics['novel_metrics']['avg_recall']:.4f}")
    print(f"Novel Avg F1: {metrics['novel_metrics']['avg_f1']:.4f}")


if __name__ == '__main__':
    main()


