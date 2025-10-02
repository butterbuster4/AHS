import json, os
from PIL import Image
from tile import Tile
from solver import solve_with_tiles
import utils
import multiprocessing as mp
from tqdm import tqdm


def MNA_evaluation(result_matrices, true_matrices):
    total_accurate_neighbors = len(true_matrices) * 12  # Each puzzle has 12 correct inner connections
    accurate_neighbors_found = 0

    # Get all the correct connections from the true individual
    correct_hori_connections = set()
    correct_vrti_connections = set()
    for matrix in true_matrices:
        for i, row in enumerate(matrix):
            for j, tile in enumerate(row):
                if tile is not None:
                    if j < len(row) - 1 and row[j+1] is not None:
                        correct_hori_connections.add((tile.index, row[j+1].index))
                    if i < len(matrix) - 1 and matrix[i+1][j] is not None:
                        correct_vrti_connections.add((tile.index, matrix[i+1][j].index))

    # Get all the connections from the result individual
    for matrix in result_matrices:
        for row in matrix:
            for i, tile in enumerate(row):
                if tile is not None and i < len(row) - 1 and row[i+1] is not None:
                    if (tile.index, row[i+1].index) in correct_hori_connections:
                        accurate_neighbors_found += 1

    for matrix in result_matrices:
        for i, row in enumerate(matrix):
            for j, tile in enumerate(row):
                if tile is not None and i < len(matrix) - 1 and matrix[i+1][j] is not None:
                    if (tile.index, matrix[i+1][j].index) in correct_vrti_connections:
                        accurate_neighbors_found += 1

    return accurate_neighbors_found / total_accurate_neighbors


def MPR_evaluation(result_matrices, true_matrices):
    perfectly_reconstructed_puzzles = 0
    matrix_mapping = utils.match_matrices(result_matrices, true_matrices)

    for r_idx, r_matrix in enumerate(result_matrices):
        if r_idx not in matrix_mapping:
            continue
        t_idx = matrix_mapping[r_idx]
        true_matrix = true_matrices[t_idx]

        if is_perfectly_reconstructed(r_matrix, true_matrix):
            perfectly_reconstructed_puzzles += 1

    return perfectly_reconstructed_puzzles / len(result_matrices)

def MABS_evaluation(result_matrices, true_matrices):
    absolute_positioned_tiles = 0
    total_tiles = len(true_matrices) * 9  # Each puzzle has 9 tiles
    matrix_mapping = utils.match_matrices(result_matrices, true_matrices)
    
    for r_idx, r_matrix in enumerate(result_matrices):
        if r_idx not in matrix_mapping:
            continue
        t_idx = matrix_mapping[r_idx]
        true_matrix = true_matrices[t_idx]

        for i, row in enumerate(r_matrix):
            for j, tile in enumerate(row):
                if tile is not None and tile.index == true_matrix[i][j].index:
                    absolute_positioned_tiles += 1

    return absolute_positioned_tiles / total_tiles

def is_perfectly_reconstructed(result_matricx, true_matricx):
    for i, row in enumerate(result_matricx):
        for j, tile in enumerate(row):
            if tile is not None and tile.index != true_matricx[i][j].index:
                return False
    return True


def load_bag(bag_dir):
    with open(os.path.join(bag_dir, "ground_truth.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    tiles = []
    for t in meta["tiles"]:
        path = os.path.join(bag_dir, t["file"])
        img = Image.open(path).convert("RGB")
        tiles.append(Tile(image=img, index=t["tile_id"]))

    return tiles, meta


def build_ground_truth_matrices(meta):
    num_puzzles = meta["bag_size"]
    matrices = [[[None, None, None] for _ in range(3)] for _ in range(num_puzzles)]
    for t in meta["tiles"]:
        puzzle_id = t["source_id"]
        row = t["row"]
        col = t["col"]
        tile_id = t["tile_id"]
        matrices[puzzle_id][row][col] = Tile(image=None, index=tile_id)
    return matrices


def evaluate_one_bag(bag_dir):
    tiles, meta = load_bag(bag_dir)
    true_matrices = build_ground_truth_matrices(meta)
    result_individual = solve_with_tiles(tiles, meta["bag_size"], true_matrices)
    result_matrices = result_individual.matrixs
    mna_score = MNA_evaluation(result_matrices, true_matrices)
    mpr_score = MPR_evaluation(result_matrices, true_matrices)
    mabs_score = MABS_evaluation(result_matrices, true_matrices)
    return bag_dir, mna_score, mpr_score, mabs_score


def evaluate_bags(bags_dir, num_workers=None):
    bag_dirs = [os.path.join(bags_dir, b) for b in os.listdir(bags_dir)]
    results = []

    with mp.Pool(processes=num_workers) as pool:
        for bag_dir, mna, mpr, mabs in tqdm(pool.imap_unordered(evaluate_one_bag, bag_dirs),
                                      total=len(bag_dirs), desc=f"Evaluating {bags_dir}"):
            print(f"Bag: {os.path.basename(bag_dir)}, MNA: {mna}, MPR: {mpr}, MABS: {mabs}")
            results.append((mna, mpr, mabs))

    average_mna_score = sum(r[0] for r in results) / len(results)
    average_mpr_score = sum(r[1] for r in results) / len(results)
    average_mabs_score = sum(r[2] for r in results) / len(results)
    return average_mna_score, average_mpr_score, average_mabs_score


if __name__ == "__main__":
    bags_dirs = ["./experiment_dataset/30_10"]
    output_file = "results.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for bags_dir in bags_dirs:
            avg_mna, avg_mpr, avg_mabs = evaluate_bags(bags_dir, num_workers=1)
            print(f"Average MNA score for {bags_dir}: {avg_mna}")
            print(f"Average MPR score for {bags_dir}: {avg_mpr}")
            print(f"Average MABS score for {bags_dir}: {avg_mabs}")
            
            # 写入txt
            f.write(f"Bag directory: {bags_dir}\n")
            f.write(f"Average MNA: {avg_mna}\n")
            f.write(f"Average MPR: {avg_mpr}\n")
            f.write(f"Average MABS: {avg_mabs}\n")
            f.write("\n")  # 每个bag之间空一行

