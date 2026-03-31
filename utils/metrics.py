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
        metrics['mAP'] = metrics['soft_mAP']
        metrics['roc_auc'] = metrics['soft_roc_auc']
        
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
        print("多标签分类指标摘要:")
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
        
        # 计算概率
        probs = sigmoid(logits)
        
        # 判断是否为dynamic k
        is_dynamic_k = (k == 'mixed')
        k_value = int(k) if not is_dynamic_k else None
        
        # 统计Top-k中包含指定类别的精确率、召回率、准确率（novel/base）
        def compute_topk_metrics(class_pairs):
            """
            Args:
                class_pairs: [(col_idx, class_id), ...]
            Returns:
                class_metrics: {class_id: {'precision': float, 'recall': float, 'accuracy': float, ...}}
                avg_metrics: {'precision': float, 'recall': float, 'accuracy': float}
            """
            class_metrics = {}
            total_tp = 0
            total_pred = 0
            total_actual = 0
            total_correct = 0
            total_samples = targets.shape[0]
            
            for col_idx, class_id in class_pairs:
                if col_idx >= logits.shape[1]:
                    continue
                
                # 计算pred_mask：判断该类别是否在Top-k中
                if is_dynamic_k:
                    # Dynamic k: 每个样本根据真实标签数确定k
                    pred_mask = np.zeros(total_samples, dtype=bool)
                    for i in range(total_samples):
                        sample_k = int(np.sum(targets[i]))
                        if sample_k == 0:
                            sample_k = 1  # 至少取top-1
                        top_k_indices = np.argsort(probs[i])[-sample_k:]
                        pred_mask[i] = col_idx in top_k_indices
                else:
                    # Fixed k: 使用预计算的top_k_preds
                    top_k_preds = np.argsort(probs, axis=1)[:, -k_value:]
                    pred_mask = np.any(top_k_preds == col_idx, axis=1)
                
                pred_count = int(np.sum(pred_mask))
                
                # Recall: 该类为真的样本
                actual_mask = targets[:, col_idx] == 1
                actual_count = int(np.sum(actual_mask))
                
                # TP: 预测且为真
                tp = int(np.sum(pred_mask & actual_mask))
                
                # Accuracy: 预测正确（TP + TN）/ 总样本数
                tn = int(np.sum((~pred_mask) & (~actual_mask)))
                accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
                
                precision = tp / pred_count if pred_count > 0 else 1
                recall = tp / actual_count if actual_count > 0 else 1
                    
                class_metrics[class_id] = {
                        'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                        'pred_count': pred_count,
                    'actual_count': actual_count,
                        'tp': tp
                    }
                
                total_tp += tp
                total_pred += pred_count
                total_actual += actual_count
                total_correct += (tp + tn)
                
            avg_precision = total_tp / total_pred if total_pred > 0 else 1
            avg_recall = total_tp / total_actual if total_actual > 0 else 1
            avg_accuracy = total_correct / (total_samples * len(class_pairs)) if len(class_pairs) > 0 else 0
            
            return class_metrics, {
                'precision': avg_precision,
                'recall': avg_recall,
                'accuracy': avg_accuracy
            }
        
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
        
        # Top-k 指标统计（Precision, Recall, Accuracy）
        novel_topk_metrics, novel_topk_avg = compute_topk_metrics(novel_class_pairs)
        base_topk_metrics, base_topk_avg = compute_topk_metrics(base_class_pairs)
        
        # Set-based指标（基于threshold）：只关注novel类的全局统计
        set_based_metrics = MultiLabelMetrics._compute_set_based_novel_metrics(
            probs, targets, novel_class_pairs, threshold
        )
        
        return {
            'class_metrics': class_metrics,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'topk_metrics': {
                'k': 'dynamic' if is_dynamic_k else k_value,
                'novel': {
                    'class_metrics': novel_topk_metrics,
                    'avg_precision': novel_topk_avg['precision'],
                    'avg_recall': novel_topk_avg['recall'],
                    'avg_accuracy': novel_topk_avg['accuracy']
                },
                'base': {
                    'class_metrics': base_topk_metrics,
                    'avg_precision': base_topk_avg['precision'],
                    'avg_recall': base_topk_avg['recall'],
                    'avg_accuracy': base_topk_avg['accuracy']
                }
            },
            'set_based_metrics': set_based_metrics
        }
    
    @staticmethod
    def _compute_set_based_novel_metrics(probs, targets, novel_class_pairs, threshold):
        """
        计算Set-based指标：基于阈值将预测视为集合，统计novel类的整体表现
        
        Args:
            probs: sigmoid概率，shape=(batch, num_classes)
            targets: 真实标签，shape=(batch, num_classes)
            novel_class_pairs: [(col_idx, class_id), ...]
            threshold: 分类阈值
            
        Returns:
            dict: {'accuracy': float, 'precision': float, 'recall': float}
        """
        if len(novel_class_pairs) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # 基于阈值的预测
        predictions = (probs >= threshold).astype(int)
        
        # 收集所有novel类的预测和真实标签
        all_novel_preds = []
        all_novel_targets = []
        
        for col_idx, _ in novel_class_pairs:
            if col_idx < probs.shape[1]:
                all_novel_preds.append(predictions[:, col_idx])
                all_novel_targets.append(targets[:, col_idx])
        
        if len(all_novel_preds) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # 展平为一维数组（视为所有novel预测的集合）
        all_novel_preds = np.concatenate(all_novel_preds)
        all_novel_targets = np.concatenate(all_novel_targets)
        
        # 计算TP, FP, FN, TN
        tp = np.sum((all_novel_preds == 1) & (all_novel_targets == 1))
        fp = np.sum((all_novel_preds == 1) & (all_novel_targets == 0))
        fn = np.sum((all_novel_preds == 0) & (all_novel_targets == 1))
        tn = np.sum((all_novel_preds == 0) & (all_novel_targets == 0))
        
        # 计算指标
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
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
        
        class_metrics = novel_metrics.get('class_metrics', {})
        topk_metrics = novel_metrics.get('topk_metrics', {})
        #k = topk_metrics.get('k')
        
        if not class_metrics:
            print(" 没有找到novel classes的指标")
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
        
        # 打印Top-k指标（Precision, Recall, Accuracy）
        def _print_topk_table(title, topk_info):
            avg_precision = topk_info.get('avg_precision', 0.0)
            avg_recall = topk_info.get('avg_recall', 0.0)
            avg_accuracy = topk_info.get('avg_accuracy', 0.0)
            
            print(f"{title:<10} P@k: {avg_precision:<8.4f} R@k: {avg_recall:<8.4f} Acc@k: {avg_accuracy:<8.4f}")
        
        _print_topk_table("Novel", topk_metrics.get('novel', {}))
        _print_topk_table("Base", topk_metrics.get('base', {}))
        
        # 打印Set-based指标（基于threshold）
        set_metrics = novel_metrics.get('set_based_metrics', {})
        if set_metrics:
            print("-" * 80)
            print(f"Novel Set-based (threshold=0.5):")
            print(f"  Accuracy: {set_metrics.get('accuracy', 0.0):.4f}  "
                  f"Precision: {set_metrics.get('precision', 0.0):.4f}  "
                  f"Recall: {set_metrics.get('recall', 0.0):.4f}")
            print(f"  TP: {set_metrics.get('tp', 0)}  FP: {set_metrics.get('fp', 0)}  "
                  f"FN: {set_metrics.get('fn', 0)}  TN: {set_metrics.get('tn', 0)}")
    
   
        
