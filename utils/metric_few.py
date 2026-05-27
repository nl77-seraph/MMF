"""
Multi-label classification metric module.
Includes metrics suitable for class-imbalanced settings.
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
    """Multi-label classification metric calculator."""
    
    @staticmethod
    def compute_metrics(logits, targets,config):
        """
        Compute metrics for multi-label classification.
        
        Args:
            logits: Model output logits, shape=(batch, num_classes)
            targets: Ground-truth labels, shape=(batch, num_classes)
            threshold: Classification threshold.
            
        Returns:
            dict: Dictionary containing metrics.
        """
        # Convert to numpy.
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        # Compute probabilities and predictions.
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
        """Print a metric summary."""
        print("Multi-label classification metric summary:")
        print("Metrics: | soft_mAP | sig_mAP | soft_roc_auc | sig_roc_auc | pk | mapk |")
        print(f"Values:  | {metrics.get('soft_mAP', 0.0):.4f}   | {metrics.get('sig_mAP', 0.0):.4f}  | {metrics.get('soft_roc_auc', 0.0):.4f}       | {metrics.get('sig_roc_auc', 0.0):.4f}      | {metrics.get('pk', 0.0):.4f} | {metrics.get('mapk', 0.0):.4f} |")

    @staticmethod
    def _compute_sample_precision(targets, predictions):
        """Compute sample-level precision."""
        sample_precision = []
        for i in range(targets.shape[0]):
            tp = np.sum(targets[i] * predictions[i])
            fp = np.sum((1 - targets[i]) * predictions[i])
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            sample_precision.append(precision)
        return np.array(sample_precision)
    
    @staticmethod
    def _compute_sample_recall(targets, predictions):
        """Compute sample-level recall."""
        sample_recall = []
        for i in range(targets.shape[0]):
            tp = np.sum(targets[i] * predictions[i])
            fn = np.sum(targets[i] * (1 - predictions[i]))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sample_recall.append(recall)
        return np.array(sample_recall)
    
    @staticmethod
    def _compute_sample_f1(targets, predictions):
        """Compute sample-level F1 score."""
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
        Find the optimal classification threshold for each class.
        
        Args:
            logits: Model output logits.
            targets: Ground-truth labels.
            metric: Optimization metric ('f1', 'precision', 'recall').
            
        Returns:
            optimal_thresholds: Optimal threshold for each class.
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
            
            if np.sum(class_targets) == 0:  # No positive samples.
                optimal_thresholds.append(0.5)
                continue
            
            # Compute the precision-recall curve.
            precision, recall, thresholds = precision_recall_curve(class_targets, class_probs)
            
            if metric == 'f1':
                f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
                best_idx = np.argmax(f1_scores)
            elif metric == 'precision':
                best_idx = np.argmax(precision)
            elif metric == 'recall':
                best_idx = np.argmax(recall)
            else:
                best_idx = len(thresholds) // 2  # Use the median by default.
            
            if best_idx < len(thresholds):
                optimal_thresholds.append(thresholds[best_idx])
            else:
                optimal_thresholds.append(0.5)
        
        return np.array(optimal_thresholds)
    
    @staticmethod
    def compute_novel_class_metrics(logits, targets, novel_classes, activated_classes=None, threshold=0.5, k='3'):
        """
        Compute per-class precision and recall for novel classes.
        
        Args:
            logits: Model output logits, shape=(batch, num_classes)
            targets: Ground-truth labels, shape=(batch, num_classes)
            novel_classes: List of novel class IDs, e.g. [60, 61, 62, ...]
            activated_classes: List of all active model classes, e.g. [0,1,2,...,79].
                             If None, logit column indices are assumed to match class IDs.
            threshold: Classification threshold; default is 0.5.
            
        Returns:
            dict: {
                'class_metrics': {class_id: {'precision': float, 'recall': float, 'f1': float}},
                'avg_precision': float,
                'avg_recall': float,
                'avg_f1': float
            }
        """
        # Convert to numpy.
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        # Compute probabilities.
        probs = sigmoid(logits)
        
        # Check whether k is dynamic.
        is_dynamic_k = (k == 'mixed')
        k_value = int(k) if not is_dynamic_k else None
        
        # Compute precision, recall, and accuracy for specified classes in Top-k (novel/base).
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
                
                # Compute pred_mask: whether this class is in Top-k.
                if is_dynamic_k:
                    # Dynamic k: each sample determines k from its true label count.
                    pred_mask = np.zeros(total_samples, dtype=bool)
                    for i in range(total_samples):
                        sample_k = int(np.sum(targets[i]))
                        if sample_k == 0:
                            sample_k = 1  # At least top-1.
                        top_k_indices = np.argsort(probs[i])[-sample_k:]
                        pred_mask[i] = col_idx in top_k_indices
                else:
                    # Fixed k: use precomputed top_k_preds.
                    top_k_preds = np.argsort(probs, axis=1)[:, -k_value:]
                    pred_mask = np.any(top_k_preds == col_idx, axis=1)
                
                pred_count = int(np.sum(pred_mask))
                
                # Recall: samples where this class is true.
                actual_mask = targets[:, col_idx] == 1
                actual_count = int(np.sum(actual_mask))
                
                # TP: predicted and true.
                tp = int(np.sum(pred_mask & actual_mask))
                
                # Accuracy: correct predictions (TP + TN) / total samples.
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
        
        # Determine column indices for novel classes in logits.
        if activated_classes is not None:
            # Map novel_classes to column indices if activated_classes is provided.
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
            # Assume logit column indices directly match class IDs.
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
                'novel_avg_precision': 0.0,
                'novel_avg_recall': 0.0,
                'novel_avg_f1': 0.0,
                'novel_pk': 0.0,
                'novel_rk': 0.0,
                'novel_acck': 0.0,
                'base_pk': 0.0,
                'base_rk': 0.0,
                'base_acck': 0.0,
                'novel_set_accuracy': 0.0,
                'novel_set_precision': 0.0,
                'novel_set_recall': 0.0,
                'topk_metrics': {
                    'k': k,
                    'novel': {
                        'class_precision': {},
                        'avg_precision': 0.0,
                        'avg_recall': 0.0,
                        'avg_accuracy': 0.0
                    },
                    'base': {
                        'class_precision': {},
                        'avg_precision': 0.0,
                        'avg_recall': 0.0,
                        'avg_accuracy': 0.0
                    }
                },
                'set_based_metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
            }
        
        # Compute metrics for each novel class.
        class_metrics = {}
        precisions = []
        recalls = []
        f1_scores = []
        
        for novel_idx, novel_class_id in zip(novel_indices, novel_classes):
            if novel_idx >= logits.shape[1]:
                continue
                
            # Get predictions and ground-truth labels for this class.
            class_probs = probs[:, novel_idx]
            class_targets = targets[:, novel_idx]
            
            # Binarize predictions.
            class_predictions = (class_probs >= threshold).astype(int)
            
            # Compute TP, FP, and FN.
            tp = np.sum((class_predictions == 1) & (class_targets == 1))
            fp = np.sum((class_predictions == 1) & (class_targets == 0))
            fn = np.sum((class_predictions == 0) & (class_targets == 1))
            
            # Compute precision, recall, and F1.
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
                'support': int(np.sum(class_targets))  # Number of true positive samples.
            }
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # Compute averages.
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        # Top-k metric statistics: precision, recall, and accuracy.
        novel_topk_metrics, novel_topk_avg = compute_topk_metrics(novel_class_pairs)
        base_topk_metrics, base_topk_avg = compute_topk_metrics(base_class_pairs)
        
        # Set-based metrics based on threshold; focus only on global novel-class statistics.
        set_based_metrics = MultiLabelMetrics._compute_set_based_novel_metrics(
            probs, targets, novel_class_pairs, threshold
        )

        flat_novel_summary = {
            'novel_avg_precision': avg_precision,
            'novel_avg_recall': avg_recall,
            'novel_avg_f1': avg_f1,
            'novel_pk': novel_topk_avg['precision'],
            'novel_rk': novel_topk_avg['recall'],
            'novel_acck': novel_topk_avg['accuracy'],
            'base_pk': base_topk_avg['precision'],
            'base_rk': base_topk_avg['recall'],
            'base_acck': base_topk_avg['accuracy'],
            'novel_set_accuracy': set_based_metrics.get('accuracy', 0.0),
            'novel_set_precision': set_based_metrics.get('precision', 0.0),
            'novel_set_recall': set_based_metrics.get('recall', 0.0),
            'novel_set_tp': set_based_metrics.get('tp', 0),
            'novel_set_fp': set_based_metrics.get('fp', 0),
            'novel_set_fn': set_based_metrics.get('fn', 0),
            'novel_set_tn': set_based_metrics.get('tn', 0)
        }
        
        return {
            'class_metrics': class_metrics,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            **flat_novel_summary,
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
        Compute set-based metrics: treat predictions as sets based on a threshold and evaluate overall novel-class performance.
        
        Args:
            probs: Sigmoid probabilities, shape=(batch, num_classes)
            targets: Ground-truth labels, shape=(batch, num_classes)
            novel_class_pairs: [(col_idx, class_id), ...]
            threshold: Classification threshold.
            
        Returns:
            dict: {'accuracy': float, 'precision': float, 'recall': float}
        """
        if len(novel_class_pairs) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Threshold-based predictions.
        predictions = (probs >= threshold).astype(int)
        
        # Collect predictions and ground-truth labels for all novel classes.
        all_novel_preds = []
        all_novel_targets = []
        
        for col_idx, _ in novel_class_pairs:
            if col_idx < probs.shape[1]:
                all_novel_preds.append(predictions[:, col_idx])
                all_novel_targets.append(targets[:, col_idx])
        
        if len(all_novel_preds) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Flatten to one dimension and treat all novel predictions as a set.
        all_novel_preds = np.concatenate(all_novel_preds)
        all_novel_targets = np.concatenate(all_novel_targets)
        
        # Compute TP, FP, FN, and TN.
        tp = np.sum((all_novel_preds == 1) & (all_novel_targets == 1))
        fp = np.sum((all_novel_preds == 1) & (all_novel_targets == 0))
        fn = np.sum((all_novel_preds == 0) & (all_novel_targets == 1))
        tn = np.sum((all_novel_preds == 0) & (all_novel_targets == 0))
        
        # Compute metrics.
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
        Print detailed metrics for novel classes.
        
        Args:
            novel_metrics: Dictionary returned by compute_novel_class_metrics.
            novel_classes: List of novel class IDs.
        """
        print("\n" + "="*80)
        # print("Novel Classes detailed metrics (per-class precision/recall)")
        # print("="*80)
        
        class_metrics = novel_metrics.get('class_metrics', {})
        topk_metrics = novel_metrics.get('topk_metrics', {})
        #k = topk_metrics.get('k')
        
        if not class_metrics:
            print("No metrics found for novel classes")
            return
        
        # # Print table header.
        # print(f"{'Class ID':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'Support':<10}")
        # print("-" * 80)
        
        # # Print in novel_classes order.
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
        print("Novel summary: | avg_precision | avg_recall | avg_f1 | P@k | R@k | Acc@k | Set-Acc | Set-P | Set-R |")
        print(
            f"Values:        | {novel_metrics.get('novel_avg_precision', 0.0):.4f}        | "
            f"{novel_metrics.get('novel_avg_recall', 0.0):.4f}     | "
            f"{novel_metrics.get('novel_avg_f1', 0.0):.4f} | "
            f"{novel_metrics.get('novel_pk', 0.0):.4f} | "
            f"{novel_metrics.get('novel_rk', 0.0):.4f} | "
            f"{novel_metrics.get('novel_acck', 0.0):.4f}  | "
            f"{novel_metrics.get('novel_set_accuracy', 0.0):.4f}  | "
            f"{novel_metrics.get('novel_set_precision', 0.0):.4f} | "
            f"{novel_metrics.get('novel_set_recall', 0.0):.4f} |"
        )
        
        # Print Top-k metrics: precision, recall, and accuracy.
        def _print_topk_table(title, topk_info):
            avg_precision = topk_info.get('avg_precision', 0.0)
            avg_recall = topk_info.get('avg_recall', 0.0)
            avg_accuracy = topk_info.get('avg_accuracy', 0.0)
            
            print(f"{title:<10} P@k: {avg_precision:<8.4f} R@k: {avg_recall:<8.4f} Acc@k: {avg_accuracy:<8.4f}")
        
        _print_topk_table("Novel", topk_metrics.get('novel', {}))
        _print_topk_table("Base", topk_metrics.get('base', {}))
        
        # Print set-based metrics based on threshold.
        set_metrics = novel_metrics.get('set_based_metrics', {})
        if set_metrics:
            print("-" * 80)
            print(f"Novel Set-based (threshold=0.5):")
            print(f"  Accuracy: {set_metrics.get('accuracy', 0.0):.4f}  "
                  f"Precision: {set_metrics.get('precision', 0.0):.4f}  "
                  f"Recall: {set_metrics.get('recall', 0.0):.4f}")
            print(f"  TP: {set_metrics.get('tp', 0)}  FP: {set_metrics.get('fp', 0)}  "
                  f"FN: {set_metrics.get('fn', 0)}  TN: {set_metrics.get('tn', 0)}")
    
   
        
