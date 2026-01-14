#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import shutil
from pathlib import Path

def ensure_unique(dst_path: Path) -> Path:
    """若目标已存在，给文件名加 _1, _2... 后缀避免覆盖。"""
    if not dst_path.exists():
        return dst_path
    stem, suf = dst_path.stem, dst_path.suffix
    i = 1
    while True:
        alt = dst_path.with_name(f"{stem}_{i}{suf}")
        if not alt.exists():
            return alt
        i += 1

def list_files_one_class(class_dir: Path, allow_exts):
    files = [p for p in class_dir.iterdir() if p.is_file()]
    if allow_exts:
        allow = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in allow_exts}
        files = [p for p in files if p.suffix.lower() in allow]
    # 过滤隐藏文件（以 . 开头）
    files = [p for p in files if not p.name.startswith(".")]
    return files

def main():
    ap = argparse.ArgumentParser(
        description="将 OW/0..95 各类别中的样本按比例划分到 OW_train/ 与 OW_test/（保留类别子目录）。"
    )
    ap.add_argument("--ow", required=True, type=Path, help="源根目录（包含 0..95 类别子文件夹）")
    ap.add_argument("--train", required=True, type=Path, help="输出训练集目录（如 OW_train）")
    ap.add_argument("--test", required=True, type=Path, help="输出测试集目录（如 OW_test）")
    ap.add_argument("--ratio", type=float, default=0.8, help="分到训练集的比例，默认 0.8")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，默认 42（可复现）")
    ap.add_argument("--move", action="store_true", help="改为移动文件（默认复制）")
    ap.add_argument("--ext", type=str, default="", help="只处理这些后缀，逗号分隔，如: jpg,png,txt（默认全部文件）")
    ap.add_argument("--fixed", action="store_true", help="固定少数的fixed number，默认False")
    
    args = ap.parse_args()

    ow = args.ow.resolve()
    out_train = args.train.resolve()
    out_test = args.test.resolve()

    if not ow.is_dir():
        raise SystemExit(f"源目录不存在或不是目录：{ow}")

    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    # 解析扩展名过滤
    allow_exts = [s.strip() for s in args.ext.split(",") if s.strip()] if args.ext else []

    random.seed(args.seed)

    total_train = total_test = 0
    classes = [p for p in ow.iterdir() if p.is_dir()]
    # 只处理名字为数字的类别目录（0~95）
    classes = [p for p in classes if p.name.isdigit()]

    if not classes:
        raise SystemExit("在 OW 下没有找到数字命名的类别子目录（如 0,1,...,95）。")

    op = shutil.move if args.move else shutil.copy2

    for cls_dir in sorted(classes, key=lambda x: int(x.name)):
        cls_name = cls_dir.name
        files = list_files_one_class(cls_dir, allow_exts) 
        if not files:
            print(f"[跳过] 类别 {cls_name} 为空。")
            continue

        random.shuffle(files)
        k = int(len(files) * args.ratio)
        if args.fixed:
            k = 50
            train_files = files[:k]
            test_files  = files[-k:]
        else:
            train_files = files[:k]
            test_files  = files[k:]
        # 建立目标类别目录
        dst_train_cls = (out_train / cls_name)
        dst_test_cls  = (out_test / cls_name)
        dst_train_cls.mkdir(parents=True, exist_ok=True)
        dst_test_cls.mkdir(parents=True, exist_ok=True)

        # 执行复制/移动
        for src in train_files:
            dst = ensure_unique(dst_train_cls / src.name)
            op(str(src), str(dst))
        for src in test_files:
            dst = ensure_unique(dst_test_cls / src.name)
            op(str(src), str(dst))

        total_train += len(train_files)
        total_test  += len(test_files)
        print(f"[{cls_name}] -> train: {len(train_files)}, test: {len(test_files)}")

    mode = "移动" if args.move else "复制"
    print(f"\n完成。共处理类别 {len(classes)} 个。")
    print(f"{mode}到 {out_train}: {total_train} 个文件")
    print(f"{mode}到 {out_test}: {total_test} 个文件")

if __name__ == "__main__":
    main()
