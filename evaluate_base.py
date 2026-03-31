"""
Evaluate base-training model on VAL set + Reweighting Coefficients Visualization.

Usage:
  python evaluate_base.py --config base_evaluate_config.json --visualize
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

# add project root
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.meta_traffic_dataloader import MetaTrafficDataLoader
from models.feature_extractors import EnhancedMultiMetaFingerNet
from utils.metrics import MultiLabelMetrics
from utils.loss_functions import WeightedBCELoss, FocalLoss, AsymmetricLoss
from utils.misc import setup_seed


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_torch_load(ckpt_path: str, device: torch.device):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj:
            return ckpt_obj["model_state_dict"]
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"]
    return ckpt_obj


def _strip_module_prefix(state_dict: dict):
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k[7:]] = v if k.startswith("module.") else v
        if not k.startswith("module."):
            new_sd[k] = v
    # 上面写法会重复一次 key，这里简单修一下
    cleaned = {}
    for k, v in new_sd.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def build_val_loader(cfg: dict):
    val_loader = MetaTrafficDataLoader(
        query_json_path=cfg["val_query_json"],
        query_files_dir=cfg["val_query_dir"],
        support_root_dir=cfg["val_support_root_dir"],
        activated_classes=list(range(cfg["num_classes"])),
        query_target_length=cfg["query_target_length"],
        support_target_length=cfg["support_target_length"],
        shots_per_class=cfg["shots_per_class"],
        batch_size=cfg["val_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        random_sampling=True,  # 如果你想 support 固定，可改 False
    )
    return val_loader


def build_model(cfg: dict, device: torch.device):
    model = EnhancedMultiMetaFingerNet(
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
        support_blocks=cfg["support_blocks"],
        use_se_in_df=cfg.get("use_se_in_df", False),
    ).to(device)
    return model


def build_criterion(cfg: dict, device: torch.device):
    positive_ratio = cfg.get("positive_ratio", 10.0)
    pos_weight = torch.tensor([positive_ratio] * cfg["num_classes"]).to(device)

    loss_type = cfg.get("loss_type", "weighted_bce")
    if loss_type == "weighted_bce":
        return WeightedBCELoss(pos_weight=pos_weight)
    if loss_type == "focal":
        return FocalLoss(
            alpha=cfg.get("focal_alpha", 0.25),
            gamma=cfg.get("focal_gamma", 2.0),
            pos_weight=pos_weight,
        )
    if loss_type == "asy":
        return AsymmetricLoss(gamma_pos=0.0, gamma_neg=3.0, clip=0.05)

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


@torch.no_grad()
def evaluate_and_collect(model, val_loader, criterion, cfg: dict, device: torch.device, collect_weights=False):
    """
    评估模型并可选地收集dynamic_weights用于可视化
    
    Args:
        collect_weights: 是否收集dynamic_weights用于可视化
    
    Returns:
        avg_loss, metrics, collected_weights (可选)
    """
    model.eval()

    losses = []
    all_logits = []
    all_labels = []
    
    # 收集dynamic_weights用于可视化
    collected_weights = [] if collect_weights else None
    max_collect_batches = 20  # 收集20个batch（增加点数）

    for bidx, batch in enumerate(val_loader):
        query_data, support_data, support_masks, batch_info = batch

        query_data = query_data.to(device)
        support_data = support_data.to(device)
        support_masks = support_masks.to(device)
        query_labels = batch_info["query_labels"].to(device)

        results = model(query_data, support_data, support_masks)
        logits = results["logits"]

        loss = criterion(logits, query_labels.float())
        losses.append(loss.item())

        all_logits.append(logits.detach().cpu())
        all_labels.append(query_labels.detach().cpu())

        # 收集dynamic_weights
        if collect_weights and bidx < max_collect_batches:
            if "dynamic_weights" in results:
                # dynamic_weights shape: (num_classes, feature_dim)
                dw = results["dynamic_weights"].detach().cpu().numpy()
                collected_weights.append(dw)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = MultiLabelMetrics.compute_metrics(all_logits, all_labels, cfg)
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    if collect_weights:
        return avg_loss, metrics, collected_weights

    return avg_loss, metrics, None



def select_diverse_classes(num_classes, num_select=20, seed=42):
    """选择差异大的类别（均匀采样）"""
    np.random.seed(seed)
    if num_classes <= num_select:
        return list(range(num_classes))
    
    # 均匀采样
    indices = np.linspace(0, num_classes - 1, num_select, dtype=int)
    return sorted(indices.tolist())



def visualize_reweighting_coefficients(collected_weights, output_path='reweighting_visualization.png', 
                                       num_classes_display=20, num_features_display=256):
    """
    绘制reweighting coefficients的热力图和t-SNE可视化 (字体加大 + 统一配置版)
    """
    
    # ========================== 🎨 字体配置区域 (随时在此处调整) ==========================
    # 这里的数值是基于你“增加一倍”的需求设定的
    font_config = {
        'title': 26,        # 大标题字体 (原12 -> 26)
        'axis_label': 22,   # 坐标轴名称字体 (原11 -> 22)
        'tick_label': 16,   # 刻度标签(如C0, C1)字体 (原8 -> 16)
        'cbar_label': 20,   # 色条说明文字字体 (原10 -> 20)
        'annotation': 12,   # t-SNE图中的文字标注字体 (原9 -> 18)
        'legend': 18        # 图例字体 (原9 -> 18)
    }
    # ====================================================================================

    if not collected_weights or len(collected_weights) == 0:
        print(" dynamic_weights")
        return
    
    print(f"\nReweighting Coefficients...")
    
    # 1. 数据准备
    num_batches = len(collected_weights)
    num_classes, feature_dim = collected_weights[0].shape
    

    selected_classes = select_diverse_classes(num_classes, num_classes_display)
    
    # 2. 诊断：检查是否存在点重叠
    print(f"   - [] :")
    all_weights_np = np.array(collected_weights) 
    
    for cls_idx in selected_classes[:5]: 
        cls_vectors = all_weights_np[:, cls_idx, :]
        var = np.var(cls_vectors, axis=0).mean()
        status = "⚠️ 高度重叠 (Stacked)" if var < 1e-4 else "✅ 分散 (Spread)"
        print(f"C{cls_idx}: Avg Variance = {var:.6f} -> {status}")

    # 3. 准备绘图数据
    first_weights = collected_weights[0]
    feature_variance = np.var(first_weights, axis=0)
    top_feature_indices = np.argsort(feature_variance)[-num_features_display:]
    top_feature_indices = sorted(top_feature_indices)
    
    # 创建画布 - 增大尺寸以适应大字体 (原 16,4 -> 26, 9)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 9)) 
    
    # =================== (a) 热力图 ===================
    heatmap_data = first_weights[selected_classes, :][:, top_feature_indices]
    heatmap_min, heatmap_max = heatmap_data.min(), heatmap_data.max()
    if heatmap_max > heatmap_min:
        heatmap_data = (heatmap_data - heatmap_min) / (heatmap_max - heatmap_min)
    
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#fefefe', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    cmap = mcolors.LinearSegmentedColormap.from_list('red_blue_gradient', colors, N=256)
    
    im = ax1.imshow(heatmap_data, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    
    # 应用字体配置
    ax1.set_xlabel('Feature Channels (ranked by variance)', fontsize=font_config['axis_label'])
    ax1.set_ylabel('Class ID', fontsize=font_config['axis_label'])
    ax1.set_title('(a) Reweighting Coefficients Heatmap', fontsize=font_config['title'], fontweight='bold', pad=15)
    
    ax1.set_yticks(range(len(selected_classes)))
    ax1.set_yticklabels([f'C{i}' for i in selected_classes], fontsize=font_config['tick_label'])
    # 设置X轴刻度大小
    ax1.tick_params(axis='x', labelsize=font_config['tick_label'])

    # 色条
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Coefficient Value', fontsize=font_config['cbar_label'])
    cbar.ax.tick_params(labelsize=font_config['tick_label']) # 色条刻度也加大
    
    # =================== (b) t-SNE可视化 ===================
    tsne_data = []
    tsne_labels = []
    num_batches_tsne = min(20, num_batches)
    
    for cls_id in selected_classes:
        for b_idx in range(num_batches_tsne):
            tsne_data.append(collected_weights[b_idx][cls_id])
            tsne_labels.append(cls_id)
            
    tsne_data = np.array(tsne_data)
    tsne_labels = np.array(tsne_labels)
    
    print(f"   - t-SNE (: {len(tsne_data)})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_data)//2))
    tsne_coords = tsne.fit_transform(tsne_data)
    
    x_span = tsne_coords[:, 0].max() - tsne_coords[:, 0].min()
    y_span = tsne_coords[:, 1].max() - tsne_coords[:, 1].min()
    jitter_scale_x = x_span * 0.015 
    jitter_scale_y = y_span * 0.015
    
    cmap_classes = plt.cm.get_cmap('tab20')
    class_colors = {cls_id: cmap_classes(i / len(selected_classes)) for i, cls_id in enumerate(selected_classes)}
    
    print(f"   - ...")

    for cls_id in selected_classes:
        mask = tsne_labels == cls_id
        coords = tsne_coords[mask]
        if len(coords) == 0: continue
            
        mean_coord = coords.mean(axis=0)
        
        noise_x = np.random.normal(0, jitter_scale_x, size=coords.shape[0])
        noise_y = np.random.normal(0, jitter_scale_y, size=coords.shape[0])
        
        coords_jittered = coords.copy()
        coords_jittered[:, 0] += noise_x
        coords_jittered[:, 1] += noise_y
        
        # 1. 绘制散点 (增大点的大小以匹配大字体)
        ax2.scatter(coords_jittered[:, 0], coords_jittered[:, 1], 
                    color=class_colors[cls_id], alpha=0.6, 
                    s=100,  # [调整] 点变大: 60 -> 100
                    edgecolors='white', linewidths=0.3, zorder=5)

        # 2. 绘制五角星
        ax2.scatter(mean_coord[0], mean_coord[1], 
                    color=class_colors[cls_id], marker='*', 
                    s=300,  # [调整] 星星变大: 180 -> 300
                    alpha=0.8, edgecolors='black', linewidths=1, zorder=10)
        
        # 3. 绘制文字标签 (应用配置字体)
        ax2.text(mean_coord[0] + jitter_scale_x*1.5, mean_coord[1] + jitter_scale_y*1.5, 
                 f'C{cls_id}', 
                 fontsize=font_config['annotation'], # 应用配置
                 color='black', alpha=0.7, verticalalignment='bottom', horizontalalignment='left',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5, edgecolor='none'),
                 zorder=15)
    
    # 应用字体配置到 Label 和 Title
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=font_config['axis_label'])
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=font_config['axis_label'])
    ax2.set_title('(b) t-SNE of Reweighting Coefficients', fontsize=font_config['title'], fontweight='bold', pad=15)
    
    # 增加刻度文字大小
    ax2.tick_params(axis='both', which='major', labelsize=font_config['tick_label'])
    ax2.grid(True, alpha=0.15)
    
    # 调整图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=12, label='Class Weight', alpha=0.6), # markersize 也加大了
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                   markersize=18, label='Class Mean Center', markeredgecolor='black', alpha=0.8)
    ]
    ax2.legend(handles=legend_elements, loc='upper right', 
               fontsize=font_config['legend'], # 应用配置
               framealpha=0.5)
    
    plt.tight_layout()
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" : {output_path}")
    plt.close()
    
    return tsne_coords, tsne_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="生成reweighting coefficients可视化")
    parser.add_argument("--output", type=str, default="reweighting_visualization.png", 
                       help="可视化输出路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_seed(cfg.get("seed", 42))

    if args.cpu or (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg.get('gpu_id', 0)}")

    print(f" Device: {device}")

    # build
    val_loader = build_val_loader(cfg)
    model = build_model(cfg, device)
    criterion = build_criterion(cfg, device)

    # load ckpt
    ckpt_path = cfg.get("ckpt_path", "")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt_path not found in config: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            
    # 处理state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 移除module.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # 处理类别数不匹配
    model_state = model.state_dict()
    loaded_keys = set(new_state_dict.keys())
    model_keys = set(model_state.keys())
    
    mismatched_keys = []
    for key in loaded_keys & model_keys:
        if new_state_dict[key].shape != model_state[key].shape:
            mismatched_keys.append(key)
            print(f"   : {key}")
            print(f"      checkpoint: {new_state_dict[key].shape}")
            print(f"      model: {model_state[key].shape}")

    filtered_state_dict = {
        k: v for k, v in new_state_dict.items() 
        if k not in mismatched_keys
    }
    
    model.load_state_dict(filtered_state_dict, strict=False)

    print(f"Checkpoint")
    print(f"      {len(filtered_state_dict)}/{len(new_state_dict)}")
    if mismatched_keys:
        print(f"      {len(mismatched_keys)}")

    # 评估
    print(" Evaluating on VAL ...")
    
    if args.visualize:
        val_loss, metrics, collected_weights = evaluate_and_collect(
            model, val_loader, criterion, cfg, device, collect_weights=True
        )
        
        # 生成可视化
        if collected_weights:
            visualize_reweighting_coefficients(
                collected_weights, 
                output_path=args.output,
                num_classes_display=20,
                num_features_display=256
            )
    else:
        val_loss, metrics, _ = evaluate_and_collect(
            model, val_loader, criterion, cfg, device, collect_weights=False
        )

    print("\n" + "=" * 60)
    print(f"VAL Loss: {val_loss:.6f}")
    MultiLabelMetrics.print_metrics_summary(metrics)
    print("=" * 60)



if __name__ == "__main__":
    main()
