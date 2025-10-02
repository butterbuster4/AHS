import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# === 配置参数 ===
# 设置随机种子以保证结果可复现
SEED = 42
image_dir = './select_image/MET/Paintings'  # 原始图像文件夹
tile_per_side = 3                       # 3x3 分块
tile_size = 96 + 8                      # 原始 tile 尺寸，96为最终尺寸，8为两侧裁边总和
gap = 4                                 # 每块四周各裁掉 4 pixels (4*2=8)
output_dir = './npy_data'               # 输出 .npy 文件夹
type = 'vrti'                     # 数据集类型
category = 'paintings'          # 数据集类别

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def extract_tiles(image, tile_size=tile_size, gap=gap):
    """
    将图像切成 3x3 块，每块裁去四周 gap 像素，返回 9 个处理后的图像块。
    """
    h, w = image.shape[:2]
    # 这个断言在主流程中由于预先调整了图像大小，所以总能通过，但作为函数独立性的保证是好的
    assert h >= tile_per_side * tile_size and w >= tile_per_side * tile_size, "图像尺寸过小，无法进行切分"
    
    tiles = []
    final_tile_dim = tile_size - 2 * gap # 应该是 96
    for row in range(tile_per_side):
        for col in range(tile_per_side):
            y1 = row * tile_size + gap
            y2 = (row + 1) * tile_size - gap
            x1 = col * tile_size + gap
            x2 = (col + 1) * tile_size - gap
            tile = image[y1:y2, x1:x2]
            # 确保切片后的尺寸正确
            assert tile.shape[0] == final_tile_dim and tile.shape[1] == final_tile_dim, "切片尺寸计算错误"
            tiles.append(tile)
    return tiles

'''
def generate_pairs(tiles):
    """
    给定9个图块, 生成 (left, right, label) 的拼图对。
    label=1 为真实上下
    相邻对, label=0 为随机不相邻对。
    此函数确保正负样本数量均衡(各6个),且负样本不重复。
    """
    pairs = []
    # 创建一个3x3的索引矩阵，方便定位
    idx_matrix = np.arange(9).reshape(3, 3)

    # 1. 生成所有正样本 (共 3*2 = 6个)
    for r in range(2):  # 遍历前两行
        for c in range(3):  # 遍历所有3列
            idx1 = idx_matrix[r, c]
            idx2 = idx_matrix[r+1, c]
            # 添加 (上边图块, 下边图块, 标签1)
            pairs.append((tiles[idx1], tiles[idx2], 1))

    # 2. 高效、不重复地生成负样本 (共6个)
    negative_pairs_generated = 0
    # 使用 set 来防止生成重复的 (i, j) 负样本索引对
    generated_indices = set() 

    while negative_pairs_generated < 6:
        # 从9个图块中随机抽取两个不同的索引
        i, j = random.sample(range(9), 2)
        
        # 如果这个组合已经生成过，则跳过
        if (i, j) in generated_indices:
            continue

        # 将一维索引转换为二维坐标以便于判断
        r1, c1 = divmod(i, 3)
        r2, c2 = divmod(j, 3)

        # 检查它们是否为真实的上下相邻对
        is_left_right_horizontal_neighbor = ( r2 - r1 == 1 and c2 == c1)

        # 如果不是真实的上下相邻对，则可以作为负样本
        if not is_left_right_horizontal_neighbor:
            pairs.append((tiles[i], tiles[j], 0))
            # 记录已生成的索引对，并更新计数器
            generated_indices.add((i, j))
            negative_pairs_generated += 1
            
    return pairs
'''

import numpy as np
import random

def generate_pairs(tiles):
    """
    给定9个图块, 生成 (left, right, label) 的拼图对。
    label=1 为真实左右相邻对, label=0 为随机不相邻对。
    此函数确保正负样本数量均衡(各6个),且负样本不重复。
    """
    pairs = []
    # 创建一个3x3的索引矩阵，方便定位
    idx_matrix = np.arange(9).reshape(3, 3)

    # 1. 生成所有正样本 (共 3*2 = 6个)
    for r in range(3):  # 遍历所有三行
        for c in range(2):  # 遍历前两列
            idx1 = idx_matrix[r, c]
            idx2 = idx_matrix[r, c + 1]
            # 添加 (左边图块, 右边图块, 标签1)
            pairs.append((tiles[idx1], tiles[idx2], 1))

    # 2. 高效、不重复地生成负样本 (共6个)
    negative_pairs_generated = 0
    # 使用 set 来防止生成重复的 (i, j) 负样本索引对
    generated_indices = set() 

    while negative_pairs_generated < 6:
        # 从9个图块中随机抽取两个不同的索引
        i, j = random.sample(range(9), 2)
        
        # 如果这个组合已经生成过，则跳过
        if (i, j) in generated_indices:
            continue

        # 将一维索引转换为二维坐标以便于判断
        r1, c1 = divmod(i, 3)
        r2, c2 = divmod(j, 3)

        # 检查它们是否为真实的左右相邻对
        is_left_right_horizontal_neighbor = (r1 == r2 and c2 - c1 == 1)

        # 如果不是真实的左右相邻对，则可以作为负样本
        if not is_left_right_horizontal_neighbor:
            pairs.append((tiles[i], tiles[j], 0))
            # 记录已生成的索引对，并更新计数器
            generated_indices.add((i, j))
            negative_pairs_generated += 1
            
    return pairs


def load_all_images_and_generate_dataset():
    """
    主流程函数：加载所有图片，正确地划分数据集，处理并保存为 .npy 文件。
    """
    # 1. 首先获取所有合法的图片文件名
    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not all_files:
        print(f"错误：在目录 '{image_dir}' 中没有找到任何图片文件。")
        return
        
    # 2. 关键步骤：在图片文件层面进行分割，防止数据泄露
    # 按照 90% 训练集, 10% 验证集的比例划分
    train_files, valid_files = train_test_split(all_files, test_size=0.1, random_state=SEED)

    print(f"数据集划分完毕：{len(train_files)}个训练文件, {len(valid_files)}个验证文件。")

    def process_files_to_dataset(files, description):
        """
        一个辅助函数，用于处理一个文件列表，并返回 NumPy 格式的图像对和标签。
        """
        all_pairs = []
        used_couple = set()  # 用于存储已使用的图块对，避免重复
        # 使用tqdm显示处理进度
        for filename in tqdm(files, desc=description):
            path = os.path.join(image_dir, filename)
            img = cv2.imread(path)
            if img is None:
                print(f"警告：无法读取文件 {filename}，已跳过。")
                continue
            
            # --- 图像预处理 ---
            h, w = img.shape[:2]
            # 裁剪为中心正方形
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            img_cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
            # 缩放到目标拼接尺寸
            target_size = tile_per_side * tile_size
            img_resized = cv2.resize(img_cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
            
            # --- 生成单张图片内部数据对 ---
            tiles = extract_tiles(img_resized)
            pairs = generate_pairs(tiles)
            all_pairs.extend(pairs)
            
            # --- 生成外部负样本 ---
            # 获取其他图片的图块
            generated_outsider_pairs_count = 0
            other_files = [f for f in files if f != filename]
            while generated_outsider_pairs_count < 16:
                other_filename = random.choice(other_files)
                other_path = os.path.join(image_dir, other_filename)
                other_img = cv2.imread(other_path)
                if other_img is None:
                    print(f"警告：无法读取文件 {other_filename}，已跳过。")
                    continue

                # --- 图像预处理 ---
                h, w = other_img.shape[:2]
                # 裁剪为中心正方形
                min_dim = min(h, w)
                start_x = (w - min_dim) // 2
                start_y = (h - min_dim) // 2
                other_img_cropped = other_img[start_y:start_y + min_dim, start_x:start_x + min_dim]
                other_img_resized = cv2.resize(other_img_cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
                
                # --- 生成外部负样本 ---
                other_tiles = extract_tiles(other_img_resized)
                '''
                # 从当前图片和外部图片中分别随机选一个图块
                tile_from_current_img = random.choice(tiles)
                tile_from_other_img = random.choice(other_tiles)
                # 获取两个图块对应的index
                idx1 = tiles.index(tile_from_current_img)
                idx2 = other_tiles.index(tile_from_other_img)
                '''
                # 随机生成两个图块的索引
                idx1 = random.randint(0, len(tiles) - 1)
                idx2 = random.randint(0, len(other_tiles) - 1)
                # 获取对应的图块
                tile_from_current_img = tiles[idx1]
                tile_from_other_img = other_tiles[idx2]
                # 添加 (当前图片图块, 外部图片图块, 标签0) 左右顺序随机
                if random.random() > 0.5:
                    out_pair = (tile_from_current_img, tile_from_other_img, 0)
                    pair_identifier = ((filename, idx1), (other_filename, idx2))
                else:
                    out_pair = (tile_from_other_img, tile_from_current_img, 0)
                    pair_identifier = ((other_filename, idx2), (filename, idx1))
                # 检查是否已存在相同的对
                if pair_identifier not in used_couple:
                    all_pairs.append(out_pair)
                    generated_outsider_pairs_count += 1
                    used_couple.add(pair_identifier)
                

        if not all_pairs:
            return np.array([]), np.array([])

        # --- 数据格式化 ---
        # 分离成三个独立的列表
        lefts = [pair[0] for pair in all_pairs]
        rights = [pair[1] for pair in all_pairs]
        labels = [pair[2] for pair in all_pairs]
        
        # 将列表转换为 NumPy 数组
        lefts_np = np.array(lefts, dtype=np.uint8)
        rights_np = np.array(rights, dtype=np.uint8)
        labels_np = np.array(labels, dtype=np.uint8)
        
        # 组合成 (N, 2, H, W, C) 的形状
        images_np = np.stack([lefts_np, rights_np], axis=1)
        
        return images_np, labels_np

    # 4. 分别为训练、验证、测试集生成数据
    train_imgs, y_train = process_files_to_dataset(train_files, "生成训练数据")
    valid_imgs, y_valid = process_files_to_dataset(valid_files, "生成验证数据")
    
    # 5. 保存处理好的数据集
    print("\n正在保存 .npy 文件...")
    # 训练集
    np.save(os.path.join(output_dir, f'{type}_{category}_train_img_gap_4.npy'), train_imgs)
    np.save(os.path.join(output_dir, f'{type}_{category}_train_label_gap_4.npy'), y_train)
    # 验证集
    np.save(os.path.join(output_dir, f'{type}_{category}_valid_img_gap_4.npy'), valid_imgs)
    np.save(os.path.join(output_dir, f'{type}_{category}_valid_label_gap_4.npy'), y_valid)
    
    print(f"所有文件已成功保存在 '{output_dir}' 目录中。")
    print(f"训练集形状: {train_imgs.shape}, 标签数: {len(y_train)}")
    print(f"验证集形状: {valid_imgs.shape}, 标签数: {len(y_valid)}")
    
    print("训练集正样本数量：", np.sum(y_train == 1))
    print("训练集负样本数量：", np.sum(y_train == 0))



# --- 执行主流程 ---
if __name__ == "__main__":
    type = 'hori'
    image_dir = './select_image/MET/Paintings'  # 原始图像文件夹
    category = 'paintings'
    load_all_images_and_generate_dataset()
    
    image_dir = './select_image/MET/Artifacts'  # 原始图像文件夹
    category = 'artifacts'
    load_all_images_and_generate_dataset()
    
    image_dir = './select_image/MET/Engravings'  # 原始图像文件夹
    category = 'engravings'
    load_all_images_and_generate_dataset()