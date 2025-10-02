import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# === 配置参数 ===
# 设置随机种子以保证结果可复现
SEED = 42
IMAGE_DIR = './select_image/MET/Paintings'  # 原始图像文件夹
TILE_PER_SIDE = 3                       # 分块块数
TILE_SIZE = 96                      # 最终有用的图块尺寸
GAP = 4                                # gap
OUTPUT_DIR = './npy_data'               # 输出 .npy 文件夹
TYPE = 'vrti'                     # 数据集类型
CATEGORY = 'paintings'          # 数据集类别
INNER_NEG = 6                        # 每张图内部负样本数
OUTER_NEG = 6                        # 每张图外部负样本数

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_tiles(image, tile_per_side, tile_size, gap):
    """
    按照 JPwLEG 的方式切图：
    - 每个 tile 是 tile_size x tile_size (不裁边)
    - tile 之间用 gap 分隔
    - 每个 tile 可以随机扰动 ±perturb
    """
    h, w = image.shape[:2]
    expected_size = tile_per_side * tile_size + (tile_per_side - 1) * gap
    assert h >= expected_size and w >= expected_size, f"图像大小必须是 {expected_size}x{expected_size}, 但得到 {h}x{w}"

    tiles = []
    for row in range(tile_per_side):
        for col in range(tile_per_side):
            y1 = row * tile_size + gap
            y2 = (row + 1) * tile_size - gap
            x1 = col * tile_size + gap
            x2 = (col + 1) * tile_size - gap

            tile = image[y1:y2, x1:x2]
            assert tile.shape[0] == tile_size and tile.shape[1] == tile_size, "tile 尺寸错误"
            tiles.append(tile)

    return tiles


def generate_pairs_hori(tiles):
    """
    左右
    """
    pairs = []
    idx_matrix = np.arange(TILE_PER_SIDE * TILE_PER_SIDE).reshape(TILE_PER_SIDE, TILE_PER_SIDE)

    # 正样本
    for r in range(TILE_PER_SIDE):
        for c in range(TILE_PER_SIDE - 1):
            idx1 = idx_matrix[r, c]
            idx2 = idx_matrix[r, c + 1]
            pairs.append((tiles[idx1], tiles[idx2], 1))

    # 负样本
    negative_pairs_generated = 0
    generated_indices = set() 

    while negative_pairs_generated < INNER_NEG:
        i, j = random.sample(range(TILE_PER_SIDE * TILE_PER_SIDE), 2)
        if (i, j) in generated_indices:
            continue
        r1, c1 = divmod(i, TILE_PER_SIDE)
        r2, c2 = divmod(j, TILE_PER_SIDE)
        is_left_right_neighbor = (r1 == r2 and c2 - c1 == 1)
        if not is_left_right_neighbor:
            pairs.append((tiles[i], tiles[j], 0))
            generated_indices.add((i, j))
            negative_pairs_generated += 1
            
    return pairs

def generate_pairs_vrti(tiles):
    """
    上下
    """
    pairs = []
    idx_matrix = np.arange(TILE_PER_SIDE * TILE_PER_SIDE).reshape(TILE_PER_SIDE, TILE_PER_SIDE)

    # 正样本
    for r in range(TILE_PER_SIDE - 1):
        for c in range(TILE_PER_SIDE):
            idx1 = idx_matrix[r, c]
            idx2 = idx_matrix[r + 1, c]
            pairs.append((tiles[idx1], tiles[idx2], 1))

    # 负样本
    negative_pairs_generated = 0
    generated_indices = set() 

    while negative_pairs_generated < INNER_NEG:
        i, j = random.sample(range(TILE_PER_SIDE * TILE_PER_SIDE), 2)
        if (i, j) in generated_indices:
            continue
        r1, c1 = divmod(i, TILE_PER_SIDE)
        r2, c2 = divmod(j, TILE_PER_SIDE)
        is_left_right_neighbor = (c1 == c2 and r2 - r1 == 1)
        if not is_left_right_neighbor:
            pairs.append((tiles[i], tiles[j], 0))
            generated_indices.add((i, j))
            negative_pairs_generated += 1
            
    return pairs


def process_files_to_dataset(files, description, save_prefix):
    """
    边生成边保存，避免一次性占满内存
    """
    used_couple = set()
    all_labels = []
    all_files_saved = []

    for idx, filename in enumerate(tqdm(files, desc=description)):
        path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"警告：无法读取文件 {filename}，已跳过。")
            continue

        # --- 图像预处理 ---
        h, w = img.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        img_cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
        target_size = TILE_PER_SIDE * TILE_SIZE + (TILE_PER_SIDE - 1) * GAP
        img_resized = cv2.resize(img_cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # --- 内部 pairs ---
        tiles = extract_tiles(img_resized, TILE_PER_SIDE, TILE_SIZE, GAP)
        pairs = generate_pairs_vrti(tiles)

        # --- 外部负样本 ---
        generated_outsider_pairs_count = 0
        other_files = [f for f in files if f != filename]
        while generated_outsider_pairs_count < OUTER_NEG and len(other_files) > 0:
            other_filename = random.choice(other_files)
            other_path = os.path.join(IMAGE_DIR, other_filename)
            other_img = cv2.imread(other_path)
            if other_img is None:
                continue
            h, w = other_img.shape[:2]
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            other_img_cropped = other_img[start_y:start_y + min_dim, start_x:start_x + min_dim]
            other_img_resized = cv2.resize(other_img_cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
            other_tiles = extract_tiles(other_img_resized, TILE_PER_SIDE, TILE_SIZE, GAP)

            idx1 = random.randint(0, len(tiles) - 1)
            idx2 = random.randint(0, len(other_tiles) - 1)
            tile_from_current = tiles[idx1]
            tile_from_other = other_tiles[idx2]

            if random.random() > 0.5:
                out_pair = (tile_from_current, tile_from_other, 0)
                pair_identifier = ((filename, idx1), (other_filename, idx2))
            else:
                out_pair = (tile_from_other, tile_from_current, 0)
                pair_identifier = ((other_filename, idx2), (filename, idx1))

            if pair_identifier not in used_couple:
                pairs.append(out_pair)
                used_couple.add(pair_identifier)
                generated_outsider_pairs_count += 1

        # --- 保存当前图片的数据对 ---
        lefts = [p[0] for p in pairs]
        rights = [p[1] for p in pairs]
        labels = [p[2] for p in pairs]
        images_np = np.stack([lefts, rights], axis=1)

        save_path = os.path.join(OUTPUT_DIR, f"{save_prefix}_{idx:05d}.npz")
        np.savez_compressed(save_path, images=images_np, labels=np.array(labels, dtype=np.uint8))
        all_files_saved.append(save_path)
        all_labels.extend(labels)

    print(f"已生成 {len(all_files_saved)} 个文件，正样本 {np.sum(np.array(all_labels)==1)}，负样本 {np.sum(np.array(all_labels)==0)}")
    return all_files_saved


def merge_npz(file_list, save_prefix):
    """
    合并多个 npz 文件
    """
    all_images = []
    all_labels = []
    for f in tqdm(file_list, desc="合并文件"):
        data = np.load(f)
        all_images.append(data['images'])
        all_labels.append(data['labels'])
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    np.save(os.path.join(OUTPUT_DIR, f'{save_prefix}_img_gap_4.npy'), all_images)
    np.save(os.path.join(OUTPUT_DIR, f'{save_prefix}_label_gap_4.npy'), all_labels)
    print(f"最终保存: {all_images.shape}, {all_labels.shape}")


def load_all_images_and_generate_dataset():
    all_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not all_files:
        print(f"错误：在目录 '{IMAGE_DIR}' 中没有找到任何图片文件。")
        return
    print(f"得到：{len(all_files)}个文件")

    npz_files = process_files_to_dataset(all_files, "生成数据", f"{TYPE}_{CATEGORY}")
    print("\n正在合并 .npz 文件...")
    merge_npz(npz_files, f"{TYPE}_{CATEGORY}")


# --- 执行主流程 ---
if __name__ == "__main__":
    IMAGE_DIR = './dataset/train'
    TYPE = 'vrti'
    CATEGORY = 'train'
    load_all_images_and_generate_dataset()

    IMAGE_DIR = './dataset/valid'
    TYPE = 'vrti'
    CATEGORY = 'valid'
    load_all_images_and_generate_dataset()
