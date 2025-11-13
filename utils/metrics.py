"""
多标签分类评估指标模块
包含适合类别不均衡的评估指标
"""

import torch
import numpy as np
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve,
    roc_auc_score,
    classification_report,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')


class MultiLabelMetrics:
    """多标签分类指标计算器"""
    
    @staticmethod
    def compute_metrics(logits, targets, threshold=0.5):
        """
        计算多标签分类的各种指标
        
        Args:
            logits: 模型输出logits, shape=(batch, num_classes)
            targets: 真实标签, shape=(batch, num_classes)
            threshold: 分类阈值
            
        Returns:
            dict: 包含各种指标的字典
        """
        # 转换为numpy
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        # 计算概率和预测
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        predictions = (probs >= threshold).astype(int)
        
        metrics = {}
        
        # 1. Mean Average Precision (mAP)
        try:
            metrics['mAP'] = average_precision_score(targets, probs, average='macro')
            metrics['mAP_micro'] = average_precision_score(targets, probs, average='micro')
        except:
            metrics['mAP'] = 0.0
            metrics['mAP_micro'] = 0.0
        
        # 2. 各类别的Average Precision
        try:
            ap_scores = average_precision_score(targets, probs, average=None)
            metrics['per_class_ap'] = ap_scores.tolist() if ap_scores is not None else []
        except:
            metrics['per_class_ap'] = []
        
        # 3. Precision, Recall, F1 (macro和micro平均)
        try:
            metrics['precision_macro'] = precision_score(targets, predictions, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(targets, predictions, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
            
            metrics['precision_micro'] = precision_score(targets, predictions, average='micro', zero_division=0)
            metrics['recall_micro'] = recall_score(targets, predictions, average='micro', zero_division=0)
            metrics['f1_micro'] = f1_score(targets, predictions, average='micro', zero_division=0)
        except:
            metrics.update({
                'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                'precision_micro': 0.0, 'recall_micro': 0.0, 'f1_micro': 0.0
            })
        
        # 4. 各类别的Precision, Recall, F1
        try:
            per_class_precision = precision_score(targets, predictions, average=None, zero_division=0)
            per_class_recall = recall_score(targets, predictions, average=None, zero_division=0)
            per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
            
            metrics['per_class_precision'] = per_class_precision.tolist()
            metrics['per_class_recall'] = per_class_recall.tolist()
            metrics['per_class_f1'] = per_class_f1.tolist()
        except:
            num_classes = targets.shape[1]
            metrics['per_class_precision'] = [0.0] * num_classes
            metrics['per_class_recall'] = [0.0] * num_classes
            metrics['per_class_f1'] = [0.0] * num_classes
        
        # 5. ROC AUC（如果可能）
        try:
            metrics['roc_auc_macro'] = roc_auc_score(targets, probs, average='macro')
            metrics['roc_auc_micro'] = roc_auc_score(targets, probs, average='micro')
        except:
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_micro'] = 0.0
        
        # 6. 简化的聚合指标（用于监控）
        metrics['avg_precision'] = metrics['precision_macro']
        metrics['avg_recall'] = metrics['recall_macro']
        metrics['avg_f1'] = metrics['f1_macro']
        
        # 7. 样本级指标
        sample_precision = MultiLabelMetrics._compute_sample_precision(targets, predictions)
        sample_recall = MultiLabelMetrics._compute_sample_recall(targets, predictions)
        sample_f1 = MultiLabelMetrics._compute_sample_f1(targets, predictions)
        
        metrics['sample_precision'] = np.mean(sample_precision)
        metrics['sample_recall'] = np.mean(sample_recall)
        metrics['sample_f1'] = np.mean(sample_f1)
        
        # 8. 类别分布统计
        metrics['positive_rate'] = np.mean(targets)
        metrics['prediction_rate'] = np.mean(predictions)
        metrics['num_classes'] = targets.shape[1]
        metrics['num_samples'] = targets.shape[0]
        
        return metrics
    
    @staticmethod
    def _compute_sample_precision(targets, predictions):
        """计算样本级精确率"""
        sample_precision = []
        for i in range(targets.shape[0]):
            tp = np.sum(targets[i] * predictions[i])
            fp = np.sum((1 - targets[i]) * predictions[i])
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            sample_precision.append(precision)
        return np.array(sample_precision)
    
    @staticmethod
    def _compute_sample_recall(targets, predictions):
        """计算样本级召回率"""
        sample_recall = []
        for i in range(targets.shape[0]):
            tp = np.sum(targets[i] * predictions[i])
            fn = np.sum(targets[i] * (1 - predictions[i]))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sample_recall.append(recall)
        return np.array(sample_recall)
    
    @staticmethod
    def _compute_sample_f1(targets, predictions):
        """计算样本级F1分数"""
        sample_f1 = []
        for i in range(targets.shape[0]):
            tp = np.sum(targets[i] * predictions[i])
            fp = np.sum((1 - targets[i]) * predictions[i])
            fn = np.sum(targets[i] * (1 - predictions[i]))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            sample_f1.append(f1)
        return np.array(sample_f1)
    
    @staticmethod
    def find_optimal_thresholds(logits, targets, metric='f1'):
        """
        为每个类别找到最优分类阈值
        
        Args:
            logits: 模型输出logits
            targets: 真实标签
            metric: 优化指标 ('f1', 'precision', 'recall')
            
        Returns:
            optimal_thresholds: 每个类别的最优阈值
        """
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        probs = 1 / (1 + np.exp(-logits))
        num_classes = targets.shape[1]
        optimal_thresholds = []
        
        for class_idx in range(num_classes):
            class_targets = targets[:, class_idx]
            class_probs = probs[:, class_idx]
            
            if np.sum(class_targets) == 0:  # 没有正样本
                optimal_thresholds.append(0.5)
                continue
            
            # 计算precision-recall曲线
            precision, recall, thresholds = precision_recall_curve(class_targets, class_probs)
            
            if metric == 'f1':
                f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
                best_idx = np.argmax(f1_scores)
            elif metric == 'precision':
                best_idx = np.argmax(precision)
            elif metric == 'recall':
                best_idx = np.argmax(recall)
            else:
                best_idx = len(thresholds) // 2  # 默认选择中位数
            
            if best_idx < len(thresholds):
                optimal_thresholds.append(thresholds[best_idx])
            else:
                optimal_thresholds.append(0.5)
        
        return np.array(optimal_thresholds)
    
    @staticmethod
    def print_metrics_summary(metrics, top_k=10):
        """打印指标摘要"""
        print("📊 多标签分类指标摘要:")
        print(f"  数据统计:")
        print(f"    - 样本数: {metrics['num_samples']}")
        print(f"    - 类别数: {metrics['num_classes']}")
        print(f"    - 正样本比例: {metrics['positive_rate']:.4f}")
        print(f"    - 预测阳性比例: {metrics['prediction_rate']:.4f}")
        
        print(f"\n  主要指标:")
        print(f"    - mAP: {metrics['mAP']:.4f}")
        print(f"    - Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"    - Recall (Macro): {metrics['recall_macro']:.4f}")
        print(f"    - F1 (Macro): {metrics['f1_macro']:.4f}")
        
        print(f"\n  Micro平均:")
        print(f"    - Precision (Micro): {metrics['precision_micro']:.4f}")
        print(f"    - Recall (Micro): {metrics['recall_micro']:.4f}")
        print(f"    - F1 (Micro): {metrics['f1_micro']:.4f}")
        
        print(f"\n  样本级指标:")
        print(f"    - Sample Precision: {metrics['sample_precision']:.4f}")
        print(f"    - Sample Recall: {metrics['sample_recall']:.4f}")
        print(f"    - Sample F1: {metrics['sample_f1']:.4f}")
        
        # 显示表现最好的类别
        if 'per_class_f1' in metrics and len(metrics['per_class_f1']) > 0:
            per_class_f1 = np.array(metrics['per_class_f1'])
            best_classes = np.argsort(per_class_f1)[-top_k:][::-1]
            
            print(f"\n  表现最佳的{min(top_k, len(best_classes))}个类别:")
            for i, class_idx in enumerate(best_classes):
                f1 = per_class_f1[class_idx]
                precision = metrics['per_class_precision'][class_idx]
                recall = metrics['per_class_recall'][class_idx]
                print(f"    类别{class_idx}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")


def test_metrics():
    """测试评估指标"""
    print("测试多标签分类指标...")
    
    # 模拟数据
    batch_size = 100
    num_classes = 60
    
    # 模拟logits和targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.zeros(batch_size, num_classes)
    
    # 创建不均衡的正标签（每个样本2-4个正标签）
    for i in range(batch_size):
        num_positive = np.random.randint(2, 5)
        pos_indices = torch.randperm(num_classes)[:num_positive]
        targets[i, pos_indices] = 1.0
    
    print(f"数据形状: logits={logits.shape}, targets={targets.shape}")
    print(f"正样本比例: {targets.mean():.4f}")
    
    # 计算指标
    metrics = MultiLabelMetrics.compute_metrics(logits, targets)
    
    # 打印摘要
    MultiLabelMetrics.print_metrics_summary(metrics)
    
    # 寻找最优阈值
    print(f"\n🔍 寻找最优阈值...")
    optimal_thresholds = MultiLabelMetrics.find_optimal_thresholds(logits, targets, metric='f1')
    print(f"最优阈值范围: [{optimal_thresholds.min():.4f}, {optimal_thresholds.max():.4f}]")
    print(f"平均最优阈值: {optimal_thresholds.mean():.4f}")


if __name__ == '__main__':
    test_metrics() 