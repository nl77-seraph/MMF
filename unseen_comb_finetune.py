"""
unseen 组合 Few-shot 微调入口

特点：
- 复用 finetune.py 中的 FewshotTrainer / 分布式入口。
- 配合 generate_unseen_cross_queries.py 产生的跨区 Query 数据。
- support_set 复用 base training 的单标签/ support_data（配置文件指定）。
"""

import os
import json
import argparse
import torch

from finetune import FewshotTrainer, run_distributed_training  # 直接复用已有训练逻辑
from utils.misc import setup_seed


def load_config(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r") as f:
        cfg = json.load(f)
    required = [
        "train_query_json",
        "train_query_dir",
        "train_support_dir",
        "base_classes",
        "novel_classes",
        "k_shot",
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"配置缺少字段: {key}")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Unseen 组合 Few-shot 微调")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_seed(config.get("seed", 42))

    use_distributed = False
    gpus = config.get("gpus", [])
    if torch.cuda.is_available() and gpus:
        available = torch.cuda.device_count()
        for gpu in gpus:
            if gpu >= available:
                raise ValueError(f"GPU {gpu} 不存在，当前可用 {available} 张")
        use_distributed = len(gpus) > 1
        config["use_distributed"] = use_distributed
    else:
        config["use_distributed"] = False

    if use_distributed:
        world_size = len(config["gpus"])
        torch.multiprocessing.spawn(
            run_distributed_training,
            args=(world_size, config),
            nprocs=world_size,
            join=True,
        )
    else:
        if torch.cuda.is_available() and config.get("gpus"):
            torch.cuda.set_device(config["gpus"][0])

        trainer = FewshotTrainer(config)
        trainer.setup_data_loaders()
        trainer.setup_model()
        trainer.setup_loss_function()
        trainer.setup_optimizer()
        trainer.setup_logging()
        trainer.train()


if __name__ == "__main__":
    main()





