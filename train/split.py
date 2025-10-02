# split_dataset.py
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def collect_images(root: str) -> List[str]:
    """递归收集 root 下的所有图片绝对路径。"""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(dirpath, fn))
    return files

def ensure_parent(dirpath: str):
    os.makedirs(dirpath, exist_ok=True)

def unique_dest_flat(dest_dir: str, src_path: str, used_names: set):
    """扁平化保存，遇到同名则在文件名后加序号避免覆盖。"""
    base = os.path.basename(src_path)
    name, ext = os.path.splitext(base)
    candidate = base
    i = 1
    while candidate in used_names or os.path.exists(os.path.join(dest_dir, candidate)):
        candidate = f"{name}_{i}{ext}"
        i += 1
    used_names.add(candidate)
    return os.path.join(dest_dir, candidate)

def split_dataset(
    image_folder: str,
    output_dir: str,
    *,
    seed: int = 42,
    counts: Optional[Tuple[int, int, int]] = None,
    ratios: Tuple[float, float, float] = (0.75, 0.0833333, 0.1666667),
    preserve_tree: bool = False,
    move_files: bool = False
) -> dict:
    """
    将 image_folder 下的所有图片划分为 train/valid/test 并保存到 output_dir。

    参数:
      image_folder: 原始图片根目录（会递归）
      output_dir: 输出目录（将创建 train/ valid/ test 子目录）
      seed: 随机种子
      counts: (train_n, valid_n, test_n) 三个整数，优先级高于 ratios。如果指定则按数量划分。
      ratios: (train_ratio, valid_ratio, test_ratio)（和应为1）
      preserve_tree: 是否在输出中保留原相对路径（保持子文件夹结构）。False 时会扁平化并自动避免重名。
      move_files: 是否移动文件（True）否则复制（False，默认）

    返回:
      一个字典包含统计信息。
    """
    random.seed(seed)
    all_files = collect_images(image_folder)
    n_total = len(all_files)
    if n_total == 0:
        raise ValueError(f"No images found under {image_folder}")

    random.shuffle(all_files)

    # 计算划分数量
    if counts is not None:
        train_n, valid_n, test_n = counts
        if train_n + valid_n + test_n > n_total:
            raise ValueError(f"Requested counts {counts} exceed total images {n_total}")
    else:
        tr, vr, ter = ratios
        if abs((tr + vr + ter) - 1.0) > 1e-6:
            raise ValueError("ratios must sum to 1.0")
        train_n = int(n_total * tr)
        valid_n = int(n_total * vr)
        test_n = n_total - train_n - valid_n

    train_files = all_files[:train_n]
    valid_files = all_files[train_n:train_n + valid_n]
    test_files = all_files[train_n + valid_n:train_n + valid_n + test_n]

    # 准备输出目录
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    test_dir = os.path.join(output_dir, 'test')
    ensure_parent(train_dir); ensure_parent(valid_dir); ensure_parent(test_dir)

    def copy_list(file_list: List[str], dest_root: str):
        used_names = set()
        for src in file_list:
            if preserve_tree:
                rel = os.path.relpath(src, image_folder)
                dest_path = os.path.join(dest_root, rel)
                dest_parent = os.path.dirname(dest_path)
                ensure_parent(dest_parent)
            else:
                dest_path = unique_dest_flat(dest_root, src, used_names)

            if move_files:
                shutil.move(src, dest_path)
            else:
                shutil.copy2(src, dest_path)

    copy_list(train_files, train_dir)
    copy_list(valid_files, valid_dir)
    copy_list(test_files, test_dir)

    info = {
        'total': n_total,
        'train_n': len(train_files),
        'valid_n': len(valid_files),
        'test_n': len(test_files),
        'preserve_tree': preserve_tree,
        'move_files': move_files
    }
    return info

# ------------- Usage examples -------------
if __name__ == "__main__":
    info = split_dataset(
        image_folder="./select_image/MET",
        output_dir="./dataset",
        seed=42,
        counts=(9000, 1000, 2000),   # 注意：总数必须 <= 可用图片数
        preserve_tree=False
    )
    print("Split done:", info)
