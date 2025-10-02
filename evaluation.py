'''
    Evaluate the performance of the proposed result.
'''
from PIL import Image
import utils

def MNA_evaluation(result_matrices, true_matrices):
    """
        Evaluate Neighbor Accuracy (NA)
    """
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
    # hori
    for matrix in result_matrices:
        for row in matrix:
            for i, tile in enumerate(row):
                if tile is not None and i < len(row) - 1 and row[i+1] is not None:
                    if (tile.index, row[i+1].index) in correct_hori_connections:
                        accurate_neighbors_found += 1

    # vrti
    for matrix in result_matrices:
        for i, row in enumerate(matrix):
            for j, tile in enumerate(row):
                if tile is not None and i < len(matrix) - 1 and matrix[i+1][j] is not None:
                    if (tile.index, matrix[i+1][j].index) in correct_vrti_connections:
                        accurate_neighbors_found += 1

    return accurate_neighbors_found / total_accurate_neighbors

def MPR_evaluation(result_matrices, true_matrices):
    """
        Evaluate perfect reconstruction ratio (MPR)
    """
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

def is_perfectly_reconstructed(result_matricx, true_matricx):
    for i, row in enumerate(result_matricx):
        for j, tile in enumerate(row):
            if tile is not None and tile.index != true_matricx[i][j].index:
                return False
    return True

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