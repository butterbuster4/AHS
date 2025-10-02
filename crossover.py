import random
from torchvision import models
import numpy as np
import torch
from fitness import fitness_individual
from typing import Set, Tuple, Optional
from image_analysis import ImageAnalysis
from individual import Individual
from tile import Tile
from torchvision import transforms
from joblib import Parallel, delayed

import utils
class Crossover:
    def __init__(self, parent1: Individual, parent2: Individual, pef_model, matrix_tile_limit: int = 9):
        self.parent1 = parent1
        self.parent2 = parent2
        self.num_matrixs = len(self.parent1.matrixs)
        self.matrix_tile_limit = matrix_tile_limit
        self.pef_model = pef_model
        self.all_tiles: Set[Tile] = set(tile for group in parent1.groups for tile in group)

        self.child_matrixs = [[] for _ in range(self.num_matrixs)]
        self.child_groups = [[] for _ in range(self.num_matrixs)]
        self.taken_tiles: Set[Tile] = set()
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.alpha = 0.5  # the weight for local fitness and global fitness in combined fitness calculation

    
    def run(self):
        # Inherit the fully connected puzzle from parent1 or parent2
        parent1_fully_connected_matrixs = self._find_fully_connected_matrixs(self.parent1)
        parent2_fully_connected_matrixs = self._find_fully_connected_matrixs(self.parent2)
        
        fully_connected_matrixs = []
        for m in parent1_fully_connected_matrixs + parent2_fully_connected_matrixs:
            tiles_in_matrix = [tile for row in m for tile in row if tile is not None]

            # Check if any tile in the matrix is already taken
            if any(tile in self.taken_tiles for tile in tiles_in_matrix):
                continue  

            fully_connected_matrixs.append(m)
            for tile in tiles_in_matrix:
                self.taken_tiles.add(tile)

        for i, matrix in enumerate(fully_connected_matrixs):
            self.child_matrixs[i] = matrix
            self.child_groups[i] = [tile for row in matrix for tile in row if tile is not None]

        for matrix_idx in range(len(fully_connected_matrixs), self.num_matrixs):
            # 1. randomly choose a seed tile from unused tiles
            seed_tile = random.choice(list(self.all_tiles - self.taken_tiles))
            self.taken_tiles.add(seed_tile)
            
            temp_matrix = [[None for _ in range(9)] for _ in range(9)]
            temp_matrix[4][4] = seed_tile
            temp_group = [seed_tile]
            frontier = [seed_tile]
            tile_count = 1

            # 2. expand the group from the seed tile
            while frontier and tile_count < self.matrix_tile_limit:
                # shuffle the frontier to ensure randomness in tile selection
                random.shuffle(frontier)
                current_tile = frontier.pop(0)
                current_position = self._find_tile_position(current_tile, temp_matrix)
                # get the matrix index of the current tile in the parent1 
                parent1_matrix_idx = self._find_matrix_index(self.parent1, current_tile)
                # get the matrix index of the current tile in the parent2
                parent2_matrix_idx = self._find_matrix_index(self.parent2, current_tile)
                neighbors = self._get_candidate_neighbors(current_tile, parent1_matrix_idx, parent2_matrix_idx)

                for neighbor, direction in neighbors:
                    if neighbor in self.taken_tiles:
                        continue
                    if self._place_tile(neighbor, temp_matrix, current_position, direction):
                        tile_count += 1
                        temp_group.append(neighbor)
                        frontier.append(neighbor)
                    else:
                        continue
                    
                    if tile_count >= self.matrix_tile_limit:
                        break
                    
                if tile_count >= self.matrix_tile_limit:
                    break
            
            temp_matrix = self._crop_useful_area(temp_matrix)
            self.child_matrixs[matrix_idx] = temp_matrix
            self.child_groups[matrix_idx] = temp_group
            
        for matrix_idx in range(self.num_matrixs):
            if len(self.child_groups[matrix_idx]) < self.matrix_tile_limit:
                # Fill the remaining tiles with expansion
                self.child_matrixs[matrix_idx], self.child_groups[matrix_idx] = self._fill_remaining_tiles_with_expansion(
                    self.child_matrixs[matrix_idx], 
                    self.child_groups[matrix_idx]
                )

    def child(self) -> Individual:
        return Individual(groups=None, matrixs=self.child_matrixs)

    def _place_tile(self, neighbor_tile: Tile, matrix, current_position, direction: str):
        i, j = current_position
        if direction == 'right':
            j += 1
        elif direction == 'down':
            i += 1
        elif direction == 'left':
            j -= 1
        elif direction == 'up':
            i -= 1
        if matrix[i][j] is not None:
            return False
        matrix[i][j] = neighbor_tile
        
        # Check if the puzzle exceeds 3x3
        width, height = self._get_matrix_size(matrix)
        if width > 3 or height > 3:
            matrix[i][j] = None
            return False
        else:
            self.taken_tiles.add(neighbor_tile)
            return True
    
    def _get_matrix_size(self, matrix):
        top, bottom = None, None
        left, right = None, None

        rows = len(matrix)
        cols = len(matrix[0])

        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] is not None:
                    if top is None or i < top:
                        top = i
                    if bottom is None or i > bottom:
                        bottom = i
                    if left is None or j < left:
                        left = j
                    if right is None or j > right:
                        right = j

        if top is None:  # all None
            return 0, 0

        width = right - left + 1
        height = bottom - top + 1
        return width, height
        
    def _find_matrix_index(self, parent: Individual, tile: Tile) -> int:
        # find the index of the matrix that contains the tile
        for m_idx, matrix in enumerate(parent.matrixs):
            for row in matrix:
                if tile in row:
                    return m_idx
        return -1

    def _get_candidate_neighbors(self, tile: Tile, parent1_matrix_idx: int, parent2_matrix_idx: int):
        candidates = []

        # Check all four directions
        for direction in ['right', 'down', 'left', 'up']:
            t1_n = self._get_neighbor_from_parent(self.parent1, parent1_matrix_idx, tile, direction)
            t2_n = self._get_neighbor_from_parent(self.parent2, parent2_matrix_idx, tile, direction)
            
            # 1. Strongly consistent (same tile)
            if t1_n == t2_n and t1_n is not None:
                candidates.append((0, t1_n, direction))
            # 2. Less strongly consistent (best buddy)
            else:
                best = self._find_best_buddy(tile, direction)
                if best:
                    candidates.append((1, best, direction))
        
        candidates.sort(key=lambda x: x[0])
        return [(t, d) for _, t, d in candidates if t is not None]

    def _find_tile_position(self, tile: Tile, matrix) -> Optional[Tuple[int, int]]:
        # find the position of the tile in the parent matrices
        for i, row in enumerate(matrix):
            for j, t in enumerate(row):
                if t == tile:
                    return i, j

    def _get_neighbor_from_parent(self, parent: Individual, parent_matrix_index: int, tile: Tile, direction: str) -> Optional[Tile]:
        matrix = parent.matrixs[parent_matrix_index]
        i, j = self._find_tile_position(tile, matrix)
        if direction == 'right' and j + 1 < len(matrix[0]):
            return matrix[i][j + 1]
        elif direction == 'left' and j - 1 >= 0:
            return matrix[i][j - 1]
        elif direction == 'down' and i + 1 < len(matrix):
            return matrix[i + 1][j]
        elif direction == 'up' and i - 1 >= 0:
            return matrix[i - 1][j]
        return None
        
    def _find_best_buddy(self, tile: Tile, direction: str) -> Optional[Tile]:
        if direction == 'right':
            if tile.right and tile.right.left == tile:
                return tile.right
        elif direction == 'left':
            if tile.left and tile.left.right == tile:
                return tile.left
        elif direction == 'down':
            if tile.down and tile.down.up == tile:
                return tile.down
        elif direction == 'up':
            if tile.up and tile.up.down == tile:
                return tile.up

    def _crop_useful_area(self, matrix):
        top, bottom = None, None
        left, right = None, None

        rows = len(matrix)
        cols = len(matrix[0])

        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] is not None:
                    if top is None or i < top:
                        top = i
                    if bottom is None or i > bottom:
                        bottom = i
                    if left is None or j < left:
                        left = j
                    if right is None or j > right:
                        right = j

        if top is None:  # all None
            return []

        cropped = [row[left:right+1] for row in matrix[top:bottom+1]]
        return cropped

    def _find_fully_connected_matrixs(self, parent):
        fully_connected_matrixs = []
        for matrix in parent.matrixs:
            if None in [tile for row in matrix for tile in row]:
                continue
            if fitness_individual(matrix, ImageAnalysis.similarity_matrix_hori, ImageAnalysis.similarity_matrix_vrti) > 10.5:
                fully_connected_matrixs.append(matrix)
        return fully_connected_matrixs
    
    
    """def _find_fully_connected_matrixs(self, parent):
        fully_connected_matrixs = []
        for matrix in parent.matrixs:
            if None in [tile for row in matrix for tile in row]:
                continue
            local_fitness = fitness_individual(matrix, ImageAnalysis.similarity_matrix_hori, ImageAnalysis.similarity_matrix_vrti)
            global_fitness = self._fitness_global(matrix)
            
            local_norm = local_fitness / 12.0   # normalize local fitness to [0, 1], max local fitness is 12.0 for 3x3 matrix
            combined = self.alpha * local_norm + (1 - self.alpha) * global_fitness
            
            if combined > 0.85:  # threshold for fully connected matrix
                fully_connected_matrixs.append(matrix)
        return fully_connected_matrixs """
            
    def _fitness_global(self, matrix):
        # use pef_model to predict the fitness of the matrix
        if not matrix or not matrix[0]:
            return 0.0
        self.pef_model.eval()
        image = utils.create_image_from_matrix_fast(matrix, fast=True)
        image = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.pef_model(image)
            probs = torch.softmax(outputs, dim=1)  # [batch, 2]
            assembled_prob = probs[0][1].item()    # the probability of being assembled 
            
        return assembled_prob
        
        

    def _fill_remaining_tiles_with_expansion(self, matrix, group):  
        # Expand the matrix to ensure it has enough space
        matrix = self.expand_matrix(matrix)   
        while len(group) < self.matrix_tile_limit:
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    if matrix[i][j] is None:
                        continue
                    # find most possible tile from unused_tiles
                    # right
                    if matrix[i][j+1] is None:
                        best = self._from_remaining_tiles(matrix[i][j], 'right')
                        if self._place_tile(best, matrix, (i, j), 'right'): 
                            group.append(best)
                    # left
                    if matrix[i][j-1] is None:
                        best = self._from_remaining_tiles(matrix[i][j], 'left')
                        if self._place_tile(best, matrix, (i, j), 'left'):
                            group.append(best)
                    # down
                    if matrix[i+1][j] is None:
                        best = self._from_remaining_tiles(matrix[i][j], 'down')
                        if self._place_tile(best, matrix, (i, j), 'down'):
                            group.append(best)
                    # up
                    if matrix[i-1][j] is None:
                        best = self._from_remaining_tiles(matrix[i][j], 'up')
                        if self._place_tile(best, matrix, (i, j), 'up'):
                            group.append(best)  
        matrix = self._crop_useful_area(matrix) 
        return matrix, group
                        
                        
    def _from_remaining_tiles(self, tile, direction):
        remaining_tiles = self.all_tiles - self.taken_tiles

        if direction == 'right':
            similarities = ImageAnalysis.similarity_matrix_hori[tile.index]
        elif direction == 'left':
            similarities = ImageAnalysis.similarity_matrix_hori[:, tile.index]
        elif direction == 'down':
            similarities = ImageAnalysis.similarity_matrix_vrti[tile.index]
        elif direction == 'up':
            similarities = ImageAnalysis.similarity_matrix_vrti[:, tile.index]
        else:
            raise ValueError(f"Unknown direction: {direction}")

        best_tile = None
        best_score = float('-inf')
        for candidate in remaining_tiles:
            score = similarities[candidate.index]
            if score > best_score:
                best_score = score
                best_tile = candidate

        return best_tile   
    
    def expand_matrix(self, matrix, pad=2):
        rows = len(matrix)
        cols = len(matrix[0])

        new_rows = rows + 2 * pad
        new_cols = cols + 2 * pad

        new_matrix = [[None for _ in range(new_cols)] for _ in range(new_rows)]

        for i in range(rows):
            for j in range(cols):
                new_matrix[i + pad][j + pad] = matrix[i][j]

        return new_matrix
                    
                        
    
    