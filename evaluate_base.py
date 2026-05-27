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
    # The code above may duplicate a key, so clean it up here.
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
        random_sampling=True,  # Set to False if fixed support sampling is preferred.
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
    Returns:
        avg_loss, metrics
    """
    model.eval()

    losses = []
    all_logits = []
    all_labels = []

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



    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = MultiLabelMetrics.compute_metrics(all_logits, all_labels, cfg)
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0



    return avg_loss, metrics, None







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_seed(cfg.get("seed", 42))

    if args.cpu or (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg.get('gpu_id', 0)}")

    print(f"Device: {device}")

    # build
    val_loader = build_val_loader(cfg)
    model = build_model(cfg, device)
    criterion = build_criterion(cfg, device)

    # load ckpt
    ckpt_path = args.ckpt_path
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt_path not found in config: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            
    # Process state_dict.
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove the module. prefix.
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Handle class-count mismatches.
    model_state = model.state_dict()
    loaded_keys = set(new_state_dict.keys())
    model_keys = set(model_state.keys())
    
    mismatched_keys = []
    for key in loaded_keys & model_keys:
        if new_state_dict[key].shape != model_state[key].shape:
            mismatched_keys.append(key)
            print(f"  Warning: shape mismatch: {key}")
            print(f"      checkpoint: {new_state_dict[key].shape}")
            print(f"      model: {model_state[key].shape}")

    filtered_state_dict = {
        k: v for k, v in new_state_dict.items() 
        if k not in mismatched_keys
    }
    
    model.load_state_dict(filtered_state_dict, strict=False)

    print(f"  Checkpoint loaded")
    print(f"     Loaded {len(filtered_state_dict)}/{len(new_state_dict)} layers")
    if mismatched_keys:
        print(f"     Skipped {len(mismatched_keys)} layers due to shape mismatch")

    # Evaluate.
    print("Evaluating on VAL ...")
    

    val_loss, metrics, _ = evaluate_and_collect(
            model, val_loader, criterion, cfg, device, collect_weights=False
        )

    print("\n" + "=" * 60)
    print(f"VAL Loss: {val_loss:.6f}")
    MultiLabelMetrics.print_metrics_summary(metrics)
    print("=" * 60)



if __name__ == "__main__":
    main()
