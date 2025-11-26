"""
多标签分类评估指标模块
包含适合类别不均衡的评估指标
"""

import torch
import numpy as np
import math
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve,
    roc_auc_score,
    classification_report,
    multilabel_confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    x = x / np.sum(x, axis=-1, keepdims=True)
    return x


class MultiLabelMetrics:
    """多标签分类指标计算器"""
    
    @staticmethod
    def compute_metrics(logits, targets,config):
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
        sig_probs = sigmoid(logits)
        soft_probs = softmax(logits)
        #predictions = (probs >= threshold).astype(int)
        
        metrics = {}
        try:
            metrics['soft_mAP'] = average_precision_score(targets, soft_probs, average='macro')
            metrics['sig_mAP'] = average_precision_score(targets, sig_probs, average='macro')
        except:
            metrics['soft_mAP'] = 0.0
            metrics['sig_mAP'] = 0.0
        try:
            metrics['soft_roc_auc'] = roc_auc_score(targets, soft_probs, average='macro')
            metrics['sig_roc_auc'] = roc_auc_score(targets, sig_probs, average='macro')
        except:
            metrics['soft_roc_auc'] = 0.0
            metrics['sig_roc_auc'] = 0.0
        if config['tabs'] == 'mixed':
            metrics['pk'] = MultiLabelMetrics.precision_at_dynamic_k(targets, soft_probs)
            metrics['mapk'] = 0.0
        else:
            metrics['pk'] = MultiLabelMetrics.precision_at_k(targets, soft_probs, int(config['tabs']))
            metrics['mapk'] = MultiLabelMetrics.average_precision_at_k(targets, soft_probs, int(config['tabs']))

        
        return metrics
    
    @staticmethod
    def precision_at_k(y_true, y_pred, k):
        top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
        precisions = []

        for i in range(y_true.shape[0]):
            true_positives = np.sum(y_true[i, top_k_preds[i]])
            precisions.append(true_positives / k)

        return np.mean(precisions)
    @staticmethod
    def precision_at_dynamic_k(y_true, y_pred):
        precisions = []
        for i in range(y_true.shape[0]):
            k = int(np.sum(y_true[i]))
            if k == 0:
                precisions.append(1)
                continue
            top_k_preds = np.argsort(y_pred[i])[-k:] 
            true_positives = np.sum(y_true[i, top_k_preds])
            precisions.append(true_positives / k)
        return np.mean(precisions) if precisions else 0.0
    @staticmethod
    def average_precision_at_k(y_true, y_pred, k):
        res = 0
        for i in range(k):
            res += MultiLabelMetrics.precision_at_k(y_true, y_pred, i+1)
        res /= k
        return res

    @staticmethod
    def print_metrics_summary(metrics):
        """打印指标摘要"""
        print("📊 多标签分类指标摘要:")
        print("Metrics: | soft_mAP | sig_mAP | soft_roc_auc | sig_roc_auc | pk | mapk |")
        print(f"Values:  | {metrics.get('soft_mAP', 0.0):.4f}   | {metrics.get('sig_mAP', 0.0):.4f}  | {metrics.get('soft_roc_auc', 0.0):.4f}       | {metrics.get('sig_roc_auc', 0.0):.4f}      | {metrics.get('pk', 0.0):.4f} | {metrics.get('mapk', 0.0):.4f} |")

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
    
   
        
