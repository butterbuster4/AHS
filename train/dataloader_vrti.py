import os
import cv2
import numpy as np
import random
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

tile_per_side = 3
tile_size = 96 + 8
gap = 4
target_tile_effective = tile_size - 2 * gap
target_size = tile_per_side * tile_size

train_dir = "./dataset/train"
valid_dir = "./dataset/valid"
test_dir  = "./dataset/test"
output_dir = "./npy_data"
os.makedirs(output_dir, exist_ok=True)

type_name = "vrti"  # 垂直
category = "all"

batch_size = 5000  # 每批保存数量


def extract_tiles(image, tile_size=tile_size, gap=gap):
    h, w = image.shape[:2]
    assert h == tile_per_side * tile_size and w == tile_per_side * tile_size
    tiles = []
    final_tile_dim = tile_size - 2 * gap
    for row in range(tile_per_side):
        for col in range(tile_per_side):
            y1 = row * tile_size + gap
            y2 = (row + 1) * tile_size - gap
            x1 = col * tile_size + gap
            x2 = (col + 1) * tile_size - gap
            tile = image[y1:y2, x1:x2]
            tiles.append(tile)
    return tiles


def generate_pairs_vrti(tiles):
    """生成垂直正样本 + 内部负样本"""
    pairs = []
    idx_matrix = np.arange(9).reshape(3, 3)
    # 正样本：同列上下相邻
    for c in range(3):
        for r in range(2):
            pairs.append((tiles[idx_matrix[r, c]], tiles[idx_matrix[r + 1, c]], 1))
    # 负样本
    generated = set()
    while len([p for p in pairs if p[2] == 0]) < 6:
        i, j = random.sample(range(9), 2)
        if (i, j) in generated: 
            continue
        r1, c1 = divmod(i, 3)
        r2, c2 = divmod(j, 3)
        if not (c1 == c2 and r2 - r1 == 1):
            pairs.append((tiles[i], tiles[j], 0))
            generated.add((i, j))
    return pairs


def load_all_images(split_dir):
    """一次性读入所有图片并切好 tiles"""
    all_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    cache = []
    for path in tqdm(all_files, desc=f"Load {os.path.basename(split_dir)}"):
        img = cv2.imread(path)
        if img is None: 
            continue
        h, w = img.shape[:2]
        min_dim = min(h, w)
        sx, sy = (w - min_dim) // 2, (h - min_dim) // 2
        img_cropped = img[sy:sy + min_dim, sx:sx + min_dim]
        img_resized = cv2.resize(img_cropped, (target_size, target_size))
        tiles = extract_tiles(img_resized)
        cache.append((os.path.basename(path), tiles))
    return cache


def pair_generator_vrti(cache):
    """生成器：逐个产生 vrti pairs"""
    used_couple = set()
    for fname, tiles in cache:
        # 内部对
        pairs = generate_pairs_vrti(tiles)
        for p in pairs:
            yield p

        # 跨图负样本 (12 对)
        for _ in range(12):
            other_fname, other_tiles = random.choice(cache)
            if other_fname == fname: 
                continue
            idx1, idx2 = random.randint(0, 8), random.randint(0, 8)
            if random.random() > 0.5:
                out_pair = (tiles[idx1], other_tiles[idx2], 0)
                pid = ((fname, idx1), (other_fname, idx2))
            else:
                out_pair = (other_tiles[idx2], tiles[idx1], 0)
                pid = ((other_fname, idx2), (fname, idx1))
            if pid not in used_couple:
                yield out_pair
                used_couple.add(pid)


def process_split_dir(split_dir, split_name):
    cache = load_all_images(split_dir)
    if not cache:
        print(f"[WARN] {split_name} 没有图片")
        return

    batch = []
    batch_count = 0
    for p in tqdm(pair_generator_vrti(cache), desc=f"Pairs {split_name}"):
        batch.append(p)
        if len(batch) >= batch_size:
            save_batch(batch, split_name, batch_count)
            batch = []
            batch_count += 1

    # 保存剩余 batch
    if batch:
        save_batch(batch, split_name, batch_count)


def save_batch(batch, split_name, batch_idx):
    lefts = np.array([p[0] for p in batch], dtype=np.uint8)
    rights = np.array([p[1] for p in batch], dtype=np.uint8)
    labels = np.array([p[2] for p in batch], dtype=np.uint8)
    images_np = np.stack([lefts, rights], axis=1)
    np.save(os.path.join(output_dir, f"{type_name}_{category}_{split_name}_img_batch_{batch_idx}.npy"), images_np)
    np.save(os.path.join(output_dir, f"{type_name}_{category}_{split_name}_label_batch_{batch_idx}.npy"), labels)
    print(f"[DONE] {split_name} batch {batch_idx}: pairs={len(batch)} shape={images_np.shape}")


if __name__ == "__main__":
    for split_dir, split_name in [(train_dir, "train"), (valid_dir, "valid"), (test_dir, "test")]:
        process_split_dir(split_dir, split_name)
