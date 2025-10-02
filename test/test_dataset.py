# make_mixed_bag_dataset_unique.py
import os
import math
import json
import random
from typing import List, Tuple
from PIL import Image

# 6 12 20 30

# -----------------------------
# 基础工具
# -----------------------------
def center_crop_square(img: Image.Image) -> Image.Image:
    """中心裁剪为正方形。"""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def get_closest_factors(n: int) -> Tuple[int, int]:
    """将 n 分解为最接近正方形的行列 (rows, cols), rows*cols == n。"""
    root = int(math.sqrt(n))
    for r in range(root, 0, -1):
        if n % r == 0:
            return r, n // r
    return 1, n


def list_images_recursive(folder: str) -> List[str]:
    paths = []
    for sub, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                paths.append(os.path.join(sub, f))
    return paths


def split_image_to_tiles(image_path: str, source_id: int, tile_effective_size: int = 96, gap: int = 4) -> Tuple[List[Tuple[int, int, Image.Image]], Image.Image]: 
    img = Image.open(image_path).convert("RGB") 
    img = center_crop_square(img) 
    tile_original = tile_effective_size + 2 * gap 
    img = img.resize((tile_original * 3, tile_original * 3), Image.BICUBIC) 
    tiles_info = [] 
    for r in range(3): 
        for c in range(3): 
            L = c * tile_original 
            U = r * tile_original 
            R = L + tile_original 
            D = U + tile_original 
            tile = img.crop((L, U, R, D)) 
            tile = tile.crop((gap, gap, tile_original - gap, tile_original - gap)) 
            tiles_info.append((r, c, tile)) 
    # 重建 3×3 GT 图 
    source_grid = Image.new("RGB", (tile_effective_size * 3, tile_effective_size * 3)) 
    for r in range(3): 
        for c in range(3): 
            _, _, t = tiles_info[r * 3 + c] 
            source_grid.paste(t, (c * tile_effective_size, r * tile_effective_size)) 
    return tiles_info, source_grid

# -----------------------------
# 生成单个 bag
# -----------------------------
def build_one_bag(image_paths: List[str],
                  out_dir: str,
                  bag_id,
                  bag_size: int,
                  tile_effective_size: int = 96,
                  gap: int = 4,
                  seed: int = 42,
                  make_mixed_puzzle: bool = True) -> None:
    """
    生成一个混合 bag,每张图片只用一次。
    bag_id 可以是字符串。
    """
    # bag_id 转整数用于随机种子
    seed_for_bag = seed + int(hash(str(bag_id)) % (2**32))
    rng = random.Random(seed_for_bag)
    rng.shuffle(image_paths)

    bag_dir = os.path.join(out_dir, f"bag_{bag_id}")
    tiles_dir = os.path.join(bag_dir, "tiles")
    src_dir = os.path.join(bag_dir, "sources")
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    meta = {
        "bag_id": str(bag_id),
        "bag_size": bag_size,
        "tile_effective_size": tile_effective_size,
        "gap": gap,
        "tile_original_size": tile_effective_size + 2 * gap,
        "images": [],
        "tiles": [],
        "mixed_puzzle_png": None,
        "grid_rows": None,
        "grid_cols": None,
        "tile_order_in_mixed": []
    }

    all_tiles = []
    for sid, img_path in enumerate(image_paths):
        tiles_info, source_grid = split_image_to_tiles(
            img_path, source_id=sid, tile_effective_size=tile_effective_size, gap=gap
        )

        # 保存 GT 图
        src_gt_name = f"source_{sid:03d}_grid.png"
        source_grid.save(os.path.join(src_dir, src_gt_name))

        meta["images"].append({
            "source_id": sid,
            "path": os.path.abspath(img_path),
            "source_grid": f"sources/{src_gt_name}"
        })

        for r, c, timg in tiles_info:
            all_tiles.append((sid, r, c, timg))

    # 混合 tiles
    rng.shuffle(all_tiles)

    # 保存 tiles 并记录映射
    for tid, (sid, r, c, timg) in enumerate(all_tiles):
        fname = f"tile_{tid:03d}.png"
        timg.save(os.path.join(tiles_dir, fname))
        meta["tiles"].append({
            "tile_id": tid,
            "file": f"tiles/{fname}",
            "source_id": sid,
            "row": r,
            "col": c
        })

    # 可选：拼成一张大图
    if make_mixed_puzzle:
        total = len(all_tiles)
        rows, cols = get_closest_factors(total)
        canvas = Image.new("RGB", (cols * tile_effective_size, rows * tile_effective_size), "black")
        for i in range(total):
            _, _, _, timg = all_tiles[i]
            r = i // cols
            c = i % cols
            canvas.paste(timg, (c * tile_effective_size, r * tile_effective_size))
            meta["tile_order_in_mixed"].append(i)
        canvas_path = os.path.join(bag_dir, "mixed_puzzle.png")
        canvas.save(canvas_path)
        meta["mixed_puzzle_png"] = "mixed_puzzle.png"
        meta["grid_rows"] = rows
        meta["grid_cols"] = cols

    # 保存 ground_truth.json
    with open(os.path.join(bag_dir, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✓ Bag {bag_id} saved to {bag_dir} ({len(all_tiles)} tiles from {bag_size} images)")


# -----------------------------
# 生成 dataset（每张图片只用一次）
# -----------------------------
def make_mixed_bag_dataset_repeatable(image_folder: str,
                                      out_root: str,
                                      bag_sizes: List[int],
                                      num_bags: int,
                                      tile_effective_size: int = 96,
                                      gap: int = 4,
                                      seed: int = 2025):
    """
    使用 folder 中的所有图片生成混合拼图包，
    - 允许图片在不同 bag 中重复使用
    - 每个 bag 内部不重复
    - 保证所有图片至少被使用一次
    - 可以强制生成固定数量的 puzzle(num_bags)
    """
    os.makedirs(out_root, exist_ok=True)
    all_images = list_images_recursive(image_folder)
    total_images = len(all_images)
    if total_images == 0:
        raise ValueError(f"在 {image_folder} 没找到图片。")
    print(f"Found {total_images} images in '{image_folder}'")

    rng = random.Random(seed)

    for size in bag_sizes:
        print(f"\n===> Generating {num_bags} bags for bag_size={size} (total {total_images} images)")

        used_once = all_images.copy()
        rng.shuffle(used_once)

        # 生成 num_bags 个拼图
        for i in range(num_bags):
            # 确保所有图片至少用一次
            if used_once:
                # 先从未使用过的里面取一部分
                bag_imgs = []
                while used_once and len(bag_imgs) < size:
                    bag_imgs.append(used_once.pop())
                # 如果还没够，补充随机图片（保证不重复）
                if len(bag_imgs) < size:
                    extra = rng.sample(all_images, size - len(bag_imgs))
                    bag_imgs.extend(extra)
            else:
                # 所有图片至少用过一次后，直接随机采样
                bag_imgs = rng.sample(all_images, size)

            build_one_bag(
                image_paths=bag_imgs,
                out_dir=out_root,
                bag_id=f"{size}_{i}",
                bag_size=size,
                tile_effective_size=tile_effective_size,
                gap=gap,
                seed=seed,
                make_mixed_puzzle=True
            )


if __name__ == "__main__":
    make_mixed_bag_dataset_repeatable(
        image_folder="./dataset/test",
        out_root="./experiment_dataset/30_10",
        bag_sizes=[30],
        num_bags=10,   
        tile_effective_size=96,
        gap=4,
        seed=2025
    )
