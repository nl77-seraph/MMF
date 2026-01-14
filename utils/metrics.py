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
    
    @staticmethod
    def compute_novel_class_metrics(logits, targets, novel_classes, activated_classes=None, threshold=0.5, k='3'):
        """
        计算novel classes每个类别的精确率和召回率
        
        Args:
            logits: 模型输出logits, shape=(batch, num_classes)
            targets: 真实标签, shape=(batch, num_classes)
            novel_classes: novel类别ID列表，例如 [60, 61, 62, ...]
            activated_classes: 模型激活的所有类别列表，例如 [0,1,2,...,79]
                             如果为None，则假设logits的列索引对应类别ID
            threshold: 分类阈值，默认0.5
            
        Returns:
            dict: {
                'class_metrics': {class_id: {'precision': float, 'recall': float, 'f1': float}},
                'avg_precision': float,
                'avg_recall': float,
                'avg_f1': float
            }
        """
        # 转换为numpy
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        if k == 'mixed':
            k = 3
        else:
            k = int(k)
        # 计算概率
        probs = sigmoid(logits)
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        
        # 统计Top-k中包含指定类别的精确率（novel/base）
        def compute_topk_precision(class_pairs):
            """
            Args:
                class_pairs: [(col_idx, class_id), ...]
            Returns:
                class_precisions: {class_id: {'precision': float, 'pred_count': int, 'tp': int}}
                avg_precision: float
            """
            class_precisions = {}
            total_tp = 0
            total_pred = 0
            
            for col_idx, class_id in class_pairs:
                if col_idx >= logits.shape[1]:
                    continue
                # 样本在Top-k中包含该类别即可视为该类别的预测
                if k != 0:
                    pred_mask = np.any(top_k_preds == col_idx, axis=1)
                    pred_count = int(np.sum(pred_mask))
                    tp = int(np.sum(pred_mask & (targets[:, col_idx] == 1)))
                    precision = tp / pred_count if pred_count > 0 else 1
                    
                    class_precisions[class_id] = {
                        'precision': precision,
                        'pred_count': pred_count,
                        'tp': tp
                    }
                    total_tp += tp
                    total_pred += pred_count
                
            avg_precision = total_tp / total_pred if total_pred > 0 else 1
            return class_precisions, avg_precision
        
        # 确定novel classes在logits中的列索引
        if activated_classes is not None:
            # 如果提供了activated_classes，需要映射novel_classes到列索引
            class_to_idx = {cls_id: idx for idx, cls_id in enumerate(activated_classes)}
            novel_indices = [class_to_idx[cls_id] for cls_id in novel_classes if cls_id in class_to_idx]
            novel_class_pairs = [
                (class_to_idx[cls_id], cls_id)
                for cls_id in novel_classes
                if cls_id in class_to_idx and class_to_idx[cls_id] < logits.shape[1]
            ]
            base_class_pairs = [
                (idx, cls_id)
                for idx, cls_id in enumerate(activated_classes)
                if cls_id not in novel_classes and idx < logits.shape[1]
            ]
        else:
            # 假设logits的列索引直接对应类别ID
            novel_indices = [cls_id for cls_id in novel_classes if cls_id < logits.shape[1]]
            novel_class_pairs = [(cls_id, cls_id) for cls_id in novel_indices]
            base_class_pairs = [
                (cls_id, cls_id)
                for cls_id in range(logits.shape[1])
                if cls_id not in novel_classes
            ]
        
        if len(novel_indices) == 0:
            return {
                'class_metrics': {},
                'avg_precision': 0.0,
                'avg_recall': 0.0,
                'avg_f1': 0.0,
                'topk_metrics': {
                    'k': k,
                    'novel': {
                        'class_precision': {},
                        'avg_precision': 0.0
                    },
                    'base': {
                        'class_precision': {},
                        'avg_precision': 0.0
                    }
                }
            }
        
        # 计算每个novel class的指标
        class_metrics = {}
        precisions = []
        recalls = []
        f1_scores = []
        
        for novel_idx, novel_class_id in zip(novel_indices, novel_classes):
            if novel_idx >= logits.shape[1]:
                continue
                
            # 获取该类别的预测和真实标签
            class_probs = probs[:, novel_idx]
            class_targets = targets[:, novel_idx]
            
            # 二值化预测
            class_predictions = (class_probs >= threshold).astype(int)
            
            # 计算TP, FP, FN
            tp = np.sum((class_predictions == 1) & (class_targets == 1))
            fp = np.sum((class_predictions == 1) & (class_targets == 0))
            fn = np.sum((class_predictions == 0) & (class_targets == 1))
            
            # 计算精确率、召回率、F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[novel_class_id] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'support': int(np.sum(class_targets))  # 真实正样本数
            }
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # 计算平均值
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        # Top-k 精确率统计
        novel_topk_precisions, novel_topk_avg = compute_topk_precision(novel_class_pairs)
        base_topk_precisions, base_topk_avg = compute_topk_precision(base_class_pairs)
        
        return {
            'class_metrics': class_metrics,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'topk_metrics': {
                'k': k,
                'novel': {
                    'class_precision': novel_topk_precisions,
                    'avg_precision': novel_topk_avg
                },
                'base': {
                    'class_precision': base_topk_precisions,
                    'avg_precision': base_topk_avg
                }
            }
        }
    
    @staticmethod
    def print_novel_class_metrics(novel_metrics, novel_classes):
        """
        打印novel classes的详细指标
        
        Args:
            novel_metrics: compute_novel_class_metrics返回的字典
            novel_classes: novel类别ID列表
        """
        print("\n" + "="*80)
        # print("🎯 Novel Classes 详细指标 (每个类别的精确率/召回率)")
        # print("="*80)
        
        class_metrics = novel_metrics.get('class_metrics', {})
        topk_metrics = novel_metrics.get('topk_metrics', {})
        #k = topk_metrics.get('k')
        
        if not class_metrics:
            print("⚠️ 没有找到novel classes的指标")
            return
        
        # # 打印表头
        # print(f"{'Class ID':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'Support':<10}")
        # print("-" * 80)
        
        # # 按novel_classes顺序打印
        # for class_id in novel_classes:
        #     if class_id in class_metrics:
        #         metrics = class_metrics[class_id]
        #         print(f"{class_id:<10} "
        #               f"{metrics['precision']:<12.4f} "
        #               f"{metrics['recall']:<12.4f} "
        #               f"{metrics['f1']:<12.4f} "
        #               f"{metrics['tp']:<8} "
        #               f"{metrics['fp']:<8} "
        #               f"{metrics['fn']:<8} "
        #               f"{metrics['support']:<10}")
        
        # print("-" * 80)
        print(f"{'Average':<10} "
              f"{novel_metrics['avg_precision']:<12.4f} "
              f"{novel_metrics['avg_recall']:<12.4f} "
              f"{novel_metrics['avg_f1']:<12.4f}")
        print("="*80)
        
        # 打印Top-k精确率（novel/base）
        def _print_topk_table(title, topk_info):
            class_precisions = topk_info.get('class_precision', {})
            avg_precision = topk_info.get('avg_precision', 0.0)
            if not class_precisions:
                print(f"{title}: 无Top-k统计")
                return
            
            # print(f"\n{title} Top-k 精确率{f' (k={k})' if k is not None else ''} (Top-k含该类视为预测)")
            # print(f"{'Class ID':<10} {'TopK_Prec':<12} {'Pred#':<8} {'TP':<6}")
            # print("-" * 80)
            # for class_id in sorted(class_precisions.keys()):
            #     metrics = class_precisions[class_id]
            #     print(f"{class_id:<10} "
            #           f"{metrics['precision']:<12.4f} "
            #           f"{metrics['pred_count']:<8} "
            #           f"{metrics['tp']:<6}")
            # print("-" * 80)
            print(f"{'Avg':<10} {avg_precision:<12.4f}")
        
        _print_topk_table("Novel", topk_metrics.get('novel', {}))
        _print_topk_table("Base", topk_metrics.get('base', {}))
    
   
        
