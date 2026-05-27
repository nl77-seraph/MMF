#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import shutil
from pathlib import Path

def ensure_unique(dst_path: Path) -> Path:
    """Append _1, _2, ... to the filename if the destination already exists."""
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
    # Filter hidden files that start with a dot.
    files = [p for p in files if not p.name.startswith(".")]
    return files

def main():
    ap = argparse.ArgumentParser(
        description="Split samples from each OW/0..95 class into OW_train/ and OW_test/ by ratio while preserving class subdirectories."
    )
    ap.add_argument("--ow", required=True, type=Path, help="Source root directory containing 0..95 class subfolders")
    ap.add_argument("--train", required=True, type=Path, help="Output training set directory, e.g. OW_train")
    ap.add_argument("--test", required=True, type=Path, help="Output test set directory, e.g. OW_test")
    ap.add_argument("--ratio", type=float, default=0.8, help="Ratio assigned to the training set; default is 0.8")
    ap.add_argument("--seed", type=int, default=42, help="Random seed; default is 42 for reproducibility")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying them")
    ap.add_argument("--ext", type=str, default="", help="Only process these suffixes, comma-separated, e.g. jpg,png,txt; default is all files")
    ap.add_argument("--fixed", action="store_true", help="Use a fixed small-sample count; default is False")
    
    args = ap.parse_args()

    ow = args.ow.resolve()
    out_train = args.train.resolve()
    out_test = args.test.resolve()

    if not ow.is_dir():
        raise SystemExit(f"Source directory does not exist or is not a directory: {ow}")

    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    # Parse the extension filter.
    allow_exts = [s.strip() for s in args.ext.split(",") if s.strip()] if args.ext else []

    random.seed(args.seed)

    total_train = total_test = 0
    classes = [p for p in ow.iterdir() if p.is_dir()]
    # Only process class directories with numeric names (0..95).
    classes = [p for p in classes if p.name.isdigit()]

    if not classes:
        raise SystemExit("No numerically named class subdirectories were found under OW, e.g. 0,1,...,95.")

    op = shutil.move if args.move else shutil.copy2

    for cls_dir in sorted(classes, key=lambda x: int(x.name)):
        cls_name = cls_dir.name
        files = list_files_one_class(cls_dir, allow_exts) 
        if not files:
            print(f"[Skip] Class {cls_name} is empty.")
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
        # Create destination class directories.
        dst_train_cls = (out_train / cls_name)
        dst_test_cls  = (out_test / cls_name)
        dst_train_cls.mkdir(parents=True, exist_ok=True)
        dst_test_cls.mkdir(parents=True, exist_ok=True)

        # Copy or move files.
        for src in train_files:
            dst = ensure_unique(dst_train_cls / src.name)
            op(str(src), str(dst))
        for src in test_files:
            dst = ensure_unique(dst_test_cls / src.name)
            op(str(src), str(dst))

        total_train += len(train_files)
        total_test  += len(test_files)
        print(f"[{cls_name}] -> train: {len(train_files)}, test: {len(test_files)}")

    mode = "Moved" if args.move else "Copied"
    print(f"\nDone. Processed {len(classes)} classes.")
    print(f"{mode} to {out_train}: {total_train} files")
    print(f"{mode} to {out_test}: {total_test} files")

if __name__ == "__main__":
    main()
