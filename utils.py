import math
import random
from PIL import Image, ImageDraw
import os

import numpy as np
from tile import Tile

def split_image_to_tiles(image_path, image_index, tiles_per_row=3, tile_size=96, crop_margin=4):
    # Read the image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    
    # Center crop to a square
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    
    # Resize to 3x3 tiles' total dimension 
    l = (tile_size + 2 * crop_margin)  * tiles_per_row  
    img = img.resize((l, l))
    
    tiles = []
    # Split into 3x3 blocks
    for row in range(tiles_per_row):
        for col in range(tiles_per_row):
            y1 = row * tile_size
            x1 = col * tile_size
            y2 = y1 + tile_size
            x2 = x1 + tile_size
            # Crop 4 pixels from each edge of every tile
            tile = img.crop((x1, y1, x2, y2))
            tiles.append((image_index, row, col, tile))  
    return tiles

def create_mixed_bag_puzzle(image_paths, puzzle_number, tile_per_row, output_path, tile_effective_size=96, gap=4):
    """
    Creates a mixed bag puzzle from multiple input images.
    Each image is split into 3x3 tiles, all tiles are mixed, and then arranged into a new 3x3 puzzle.

    Args:
        image_paths (list): A list of paths to the input images.
        output_path (str): The path where the final mixed puzzle image will be saved.
        tile_effective_size (int): The effective size of each tile after cropping (e.g., 96 pixels if original was 104 and margin 4). Defaults to 96.
        gap (int): The spacing in pixels between tiles in the final puzzle. Defaults to 4.

    Returns:
        PIL.Image.Image: The generated mixed puzzle image.
    """
    all_tiles = []
    # Collect tiles from all images
    for i, path in enumerate(image_paths):
        # Calculate the original tile size for split_image_to_tiles based on effective size and margin
        # Assuming crop_margin is 4, then tile_original_size = tile_effective_size + 2 * crop_margin
        # For tile_effective_size=96 and crop_margin=4, tile_original_size = 96 + 2*4 = 104
        tiles_from_image = split_image_to_tiles(path, i, tiles_per_row=tile_per_row, tile_size=tile_effective_size, crop_margin=gap)
        all_tiles.extend(tiles_from_image)
    
    if not all_tiles:
        print("No tiles were generated from the provided images. Cannot create puzzle.")
        return None

    # Shuffle all collected tiles
    random.shuffle(all_tiles)
    
    # Create the final mixed bag puzzle image from the shuffled tiles
    create_image_from_tiles(all_tiles, output_path, tile_size=tile_effective_size)
    
    return all_tiles

def create_image_from_tiles(tiles, output_path=None, tile_size=96):
    # Determine the grid size (ensure can fit all tiles and form a rectangle grid which may not be a square)
    # Dynamically calculate rows and columns for the combined puzzle
    total_pieces = len(tiles)
    # may not be a square grid, so we calculate rows and columns based on total pieces
    # e.g. may have 18 tiles, then grid could be 3 rows and 6 columns that ensure exact fit
    # obtain the cloest two factor combination of the total pieces
    grid_rows, grid_cols = get_cloest_factors(total_pieces)
    
    # Calculate the dimensions of the final puzzle image
    puzzle_width = grid_cols * tile_size
    puzzle_height = grid_rows * tile_size
    
    # Create a new blank image for the puzzle
    puzzle_img = Image.new('RGB', (puzzle_width, puzzle_height), color='black') # Background color can be changed

    # Paste the shuffled tiles onto the new image
    for i in range(grid_rows * grid_cols):
        if i < len(tiles): # Ensure we don't go out of bounds if not enough tiles
            tile = tiles[i].image if isinstance(tiles[i], Tile) else tiles[i][3]
            row = i // grid_cols
            col = i % grid_cols
            
            x_offset = col * tile_size
            y_offset = row * tile_size
            
            puzzle_img.paste(tile, (x_offset, y_offset))
        else:
            print(f"Warning: Not enough tiles to fill a {grid_rows}x{grid_cols} grid. Missing {grid_rows*grid_cols - len(tiles)} tiles.")
            break # Stop if we run out of tiles

    if output_path:
        # Save the puzzle image
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        puzzle_img.save(output_path)
    return puzzle_img

def create_image_from_matrix(matrix, output_path=None, tile_size=96):
    # Get the number of rows and columns in the matrix
    grid_rows = len(matrix)
    grid_cols = len(matrix[0]) if grid_rows > 0 else 0
    
    # Calculate the dimensions of the final puzzle image
    puzzle_width = grid_cols * tile_size
    puzzle_height = grid_rows * tile_size
    
    # Create a new blank image for the puzzle
    puzzle_img = Image.new('RGB', (puzzle_width, puzzle_height), color='black') # Background color can be changed

    # Paste the shuffled tiles onto the new image
    for row in range(grid_rows):
        for col in range(grid_cols):
            tile = matrix[row][col]
            x_offset = col * tile_size
            y_offset = row * tile_size
            
            if tile is None:
                continue

            puzzle_img.paste(tile.image, (x_offset, y_offset))

    if output_path:
        # Save the puzzle image
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        puzzle_img.save(output_path)
    return puzzle_img

def get_cloest_factors(n):
    """
    Get the closest factors of a number n.
    """
    factors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            factors.append(i)
            factors.append(n // i)
    return factors[-2:] if len(factors) > 1 else (1, n)  # Return the last two factors which are closest to each other

def split_puzzle_to_tiles(img_path, tile_size=96):
    """
    Splits a puzzle image into its constituent tiles.
    """
    # Read the image and convert to RGB
    img = Image.open(img_path).convert("RGB")
    width, height = img.size

    # Calculate the number of tiles in each dimension
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size
    
    tiles = []
    # Split into num_tiles_x * num_tiles_y blocks
    for row in range(num_tiles_y):
        for col in range(num_tiles_x):
            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size
            tile = img.crop((left, upper, right, lower))
            tiles.append(tile)

    return tiles

def divide_puzzle_into_groups_according_to_clusters(image_path, tiles, clusters, tile_size=96):
    """
    Divide a mixed-bag puzzle into groups according to clusters.
    """
    num_groups = len(set(clusters))
    groups = [[] for _ in range(num_groups)]
    for i, tile in enumerate(tiles):
        cluster_index = clusters[i]
        groups[cluster_index].append(tile)
    for i, group in enumerate(groups):
        if group:
            output_path = f"./puzzles/{os.path.splitext(image_path)[0]}_group_{i}.png"
            create_image_from_tiles(group, output_path, tile_size=tile_size)
    return groups

def create_puzzle(IMAGE_FOLDER, output_path, puzzle_number=10, tile_per_row=3, tile_effective_size=96, gap=4):
    images = [] 

    for subdir, _, files in os.walk(IMAGE_FOLDER):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(subdir, file))

    if len(images) < puzzle_number:
        raise ValueError(f"Not enough images found. Only {len(images)} images available.")
    selected_images = random.sample(images, puzzle_number)
    
    true_solution = create_mixed_bag_puzzle(selected_images, puzzle_number, tile_per_row=tile_per_row, output_path=output_path, tile_effective_size=tile_effective_size, gap=gap)
    return true_solution

def create_matrixs_from_groups_of_tiles(groups):
    matrixs = []
    for group in groups:
        grid_rows, grid_cols = get_cloest_factors(len(group))
        matrix = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        for i, tile in enumerate(group):
            row, col = divmod(i, grid_cols)
            matrix[row][col] = tile
        matrixs.append(matrix)
    return matrixs

def create_groups_from_matrixs(matrixs):
    groups = []
    for matrix in matrixs:
        group = [tile for row in matrix for tile in row if tile is not None]
        groups.append(group)
    return groups

def match_matrices(result_matrices, true_matrices):
    """
    Match each matrix in result_individual with the most similar matrix in true_individual
    based on maximum overlapping tiles.
    Ensures one-to-one mapping (greedy maximum matching).
    """
    mapping = {}
    
    # Collect all tile indices for each true matrix
    true_sets = []
    for t_idx, t_matrix in enumerate(true_matrices):
        indices = set()
        for row in t_matrix:
            for tile in row:
                if tile is not None:
                    indices.add(tile.index)
        true_sets.append((t_idx, indices))
    
    # Compute all overlaps between result and true matrices
    overlaps = []
    for r_idx, r_matrix in enumerate(result_matrices):
        r_indices = set()
        for row in r_matrix:
            for tile in row:
                if tile is not None:
                    r_indices.add(tile.index)

        for t_idx, t_indices in true_sets:
            overlap = len(r_indices & t_indices)
            overlaps.append((overlap, r_idx, t_idx))
    
    # Sort by overlap size (largest first)
    overlaps.sort(reverse=True, key=lambda x: x[0])
    
    used_r = set()
    used_t = set()
    
    # Greedy unique matching
    for overlap, r_idx, t_idx in overlaps:
        if r_idx not in used_r and t_idx not in used_t and overlap > 0:
            mapping[r_idx] = t_idx
            used_r.add(r_idx)
            used_t.add(t_idx)
    
    return mapping


def build_true_neighbors(true_matrix):
    """
    Build adjacency relationships from true_matrix.
    Each tile knows who its right and bottom neighbors are.
    """
    neighbors = {}
    for i, row in enumerate(true_matrix):
        for j, tile in enumerate(row):
            if tile is None:
                continue
            idx = tile.index
            neighbors[idx] = []
            # Right neighbor
            if j+1 < len(row) and row[j+1] is not None:
                neighbors[idx].append(row[j+1].index)
            # Bottom neighbor
            if i+1 < len(true_matrix) and true_matrix[i+1][j] is not None:
                neighbors[idx].append(true_matrix[i+1][j].index)
    return neighbors


def highlight_incorrect_tiles(result_individual, true_individual, save_path="highlighted_result.jpg"):
    """
    Directly recolor the tiles in result_individual that are incorrectly placed
    compared to true_individual (neighbor-based correctness).
    Instead of drawing on a separate overlay, we modify the tile's own image
    by tinting it red for visibility.
    """
    # Step 1: find matrix correspondence
    matrix_mapping = match_matrices(result_individual.matrixs, true_individual.matrixs)

    # Step 3: for each matrix, compare neighbors
    for r_idx, r_matrix in enumerate(result_individual.matrixs):
        t_idx = matrix_mapping[r_idx]
        true_matrix = true_individual.matrixs[t_idx]
        
        for i, row in enumerate(r_matrix):
            for j, tile in enumerate(row):
                if tile != true_matrix[i][j]:
                    # Apply red tint (increase red channel, decrease green/blue)
                    r, g, b = tile.image.split()
                    r = r.point(lambda p: min(255, int(p * 1.5)))   # boost red
                    g = g.point(lambda p: int(p * 0.5))            # reduce green
                    b = b.point(lambda p: int(p * 0.5))            # reduce blue
                    tile.image = Image.merge("RGB", (r, g, b))

    result_individual.to_image().save(save_path)
    
def find_tile_in_matrix(matrix, tile):
    for i, row in enumerate(matrix):
        for j, t in enumerate(row):
            if t == tile:
                return i, j
    return None, None
    
from PIL import ImageDraw

def highlight_incorrect_tile_borders(result_individual, true_individual, save_path="highlighted_borders_result.jpg"):
    # Step 1: match matrices between result and true
    matrix_mapping = match_matrices(result_individual.matrixs, true_individual.matrixs)

    # Step 2: iterate over matrices
    for r_idx, r_matrix in enumerate(result_individual.matrixs):
        t_idx = matrix_mapping[r_idx]
        true_matrix = true_individual.matrixs[t_idx]

        for i, row in enumerate(r_matrix):
            for j, tile in enumerate(row):
                # get tile dimensions
                w, h = tile.image.size
                draw = ImageDraw.Draw(tile.image)

                # find position of this tile in true matrix
                r, c = find_tile_in_matrix(true_matrix, tile)
                if r is None or c is None:
                    # tile not found in true matrix → mark all borders red
                    draw.rectangle([(0,0), (w-1,h-1)], outline=(255,0,0), width=2)
                    continue

                # check right neighbor
                if j < len(row) - 1 and c < len(true_matrix[0]) - 1:
                    if true_matrix[r][c+1] != r_matrix[i][j+1]:
                        draw.line([(w-1, 0), (w-1, h)], fill=(255, 0, 0), width=2)

                # check left neighbor
                if j > 0 and c > 0:
                    if true_matrix[r][c-1] != r_matrix[i][j-1]:
                        draw.line([(0, 0), (0, h)], fill=(255, 0, 0), width=2)

                # check top neighbor
                if i > 0 and r > 0:
                    if true_matrix[r-1][c] != r_matrix[i-1][j]:
                        draw.line([(0, 0), (w, 0)], fill=(255, 0, 0), width=2)

                # check bottom neighbor
                if i < len(r_matrix) - 1 and r < len(true_matrix) - 1:
                    if true_matrix[r+1][c] != r_matrix[i+1][j]:
                        draw.line([(0, h-1), (w, h-1)], fill=(255, 0, 0), width=2)

    # save the reconstructed image
    result_individual.to_image().save(save_path)

def create_image_from_matrix_fast(matrix, output_path=None, tile_size=96, fast=False):
    """
    拼接 matrix 成一张大图
    fast=True 时用 numpy 拼接（比 PIL.paste 快很多）
    """
    grid_rows = len(matrix)
    grid_cols = len(matrix[0]) if grid_rows > 0 else 0
    
    if fast:
        # 用 numpy 拼接
        row_imgs = []
        for row in matrix:
            row_tiles = []
            for tile in row:
                if tile is None:
                    arr = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                else:
                    img = tile.image
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)
                    arr = np.array(img.convert("RGB").resize((tile_size, tile_size)))
                row_tiles.append(arr)
            row_imgs.append(np.concatenate(row_tiles, axis=1))  # 横向拼接
        puzzle_arr = np.concatenate(row_imgs, axis=0)  # 纵向拼接
        puzzle_img = Image.fromarray(puzzle_arr)

    else:
        # 原始 PIL 版本
        puzzle_width = grid_cols * tile_size
        puzzle_height = grid_rows * tile_size
        puzzle_img = Image.new('RGB', (puzzle_width, puzzle_height), color='black')
        for row in range(grid_rows):
            for col in range(grid_cols):
                tile = matrix[row][col]
                x_offset = col * tile_size
                y_offset = row * tile_size
                if tile is None:
                    continue
                img = tile.image
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                puzzle_img.paste(img.convert("RGB").resize((tile_size, tile_size)), (x_offset, y_offset))

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        puzzle_img.save(output_path)

    return puzzle_img