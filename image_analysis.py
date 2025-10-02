from typing import List
import numpy as np
import torch
from tile import Tile
import utils
from tqdm import tqdm 
from combonet import ComboNet
from torchvision import transforms


class ImageAnalysis(object):
    similarity_matrix_hori: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    similarity_matrix_vrti: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    similarity_matrix_hori_top: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    similarity_matrix_vrti_top: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    sym_similarity_matrix_hori: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    sym_similarity_matrix_vrti: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    max_similarity_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    tiles: List[Tile] = []
    
    @classmethod
    def analyze_puzzle(cls, image_path, tiles, device, model_hori_path, model_vrti_path, batch_size: int = 1024):
        cls.calculate_similarity_matrix(image_path, tiles, device, model_hori_path, model_vrti_path, batch_size=batch_size)
        
        cls.similarity_matrix_hori_top = cls.keep_top_k(cls.similarity_matrix_hori, k=1)
        cls.similarity_matrix_vrti_top = cls.keep_top_k(cls.similarity_matrix_vrti, k=1)
        
        cls.similarity_matrix_hori_top = cls.resolve_conflicts(cls.similarity_matrix_hori, cls.similarity_matrix_hori_top)
        cls.similarity_matrix_vrti_top = cls.resolve_conflicts(cls.similarity_matrix_vrti, cls.similarity_matrix_vrti_top)
        
        cls.sym_similarity_matrix_hori = (cls.similarity_matrix_hori_top + cls.similarity_matrix_hori_top.T) 
        cls.sym_similarity_matrix_vrti = (cls.similarity_matrix_vrti_top + cls.similarity_matrix_vrti_top.T) 
        cls.max_similarity_matrix = np.maximum(cls.sym_similarity_matrix_hori, cls.sym_similarity_matrix_vrti)
        
        for tile in cls.tiles:
            cls.find_best_neighbors(tile)
            
    @classmethod
    def find_best_neighbors(cls, tile):
        idx = tile.index

        simi_hori = cls.similarity_matrix_hori[idx]
        tile.right = cls.tiles[simi_hori.argmax()] if simi_hori.max() > 0 else None

        simi_hori_col = cls.similarity_matrix_hori[:, idx]
        tile.left = cls.tiles[simi_hori_col.argmax()] if simi_hori_col.max() > 0 else None

        simi_vrti = cls.similarity_matrix_vrti[idx]
        tile.down = cls.tiles[simi_vrti.argmax()] if simi_vrti.max() > 0 else None

        simi_vrti_col = cls.similarity_matrix_vrti[:, idx]
        tile.up = cls.tiles[simi_vrti_col.argmax()] if simi_vrti_col.max() > 0 else None
        
    @classmethod
    def calculate_similarity_matrix(cls, image_path, tiles, device, model_hori_path, model_vrti_path, batch_size=1024):
        model_hori = ComboNet().to(device)
        model_hori.load_state_dict(torch.load(model_hori_path, map_location=device))
        model_vrti = ComboNet().to(device)
        model_vrti.load_state_dict(torch.load(model_vrti_path, map_location=device))
        model_hori.eval()
        model_vrti.eval()
        
        if not tiles:
            tiles = utils.split_puzzle_to_tiles(image_path)
            cls.tiles = [Tile(tile, i) for i, tile in enumerate(tiles)]
        else:
            cls.tiles = tiles
            tiles = [tile.image for tile in tiles]
        total_tiles = len(cls.tiles)
        
        cls.similarity_matrix_hori = np.zeros((total_tiles, total_tiles), dtype=np.float32)
        cls.similarity_matrix_vrti = np.zeros((total_tiles, total_tiles), dtype=np.float32)
        
        preprocess = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        tensor_tiles = [preprocess(tile).unsqueeze(0).to(device) for tile in tiles]
        
        pairs1, pairs2, idx_pairs = [], [], []
        for i in tqdm(range(total_tiles), desc="Calculating similarity matrix"):
            for j in range(total_tiles):
                if i == j:
                    continue
                pairs1.append(tensor_tiles[i])
                pairs2.append(tensor_tiles[j])
                idx_pairs.append((i, j))

                if len(pairs1) >= batch_size:
                    batch1 = torch.cat(pairs1, 0)
                    batch2 = torch.cat(pairs2, 0)
                    with torch.no_grad():
                        probs_h = model_hori(batch1, batch2).cpu().numpy()
                        probs_v = model_vrti(batch1, batch2).cpu().numpy()
                    for (ii, jj), ph, pv in zip(idx_pairs, probs_h, probs_v):
                        cls.similarity_matrix_hori[ii, jj] = ph
                        cls.similarity_matrix_vrti[ii, jj] = pv
                    pairs1, pairs2, idx_pairs = [], [], []
        
        if pairs1:  # flush last batch
            batch1 = torch.cat(pairs1, 0)
            batch2 = torch.cat(pairs2, 0)
            with torch.no_grad():
                probs_h = model_hori(batch1, batch2).cpu().numpy()
                probs_v = model_vrti(batch1, batch2).cpu().numpy()
            for (ii, jj), ph, pv in zip(idx_pairs, probs_h, probs_v):
                cls.similarity_matrix_hori[ii, jj] = ph
                cls.similarity_matrix_vrti[ii, jj] = pv
    
    @staticmethod
    def keep_top_k(similarity_matrix, k=1):
        threshold = np.partition(similarity_matrix, -k, axis=1)[:, -k][:, None]
        return np.where(similarity_matrix >= threshold, similarity_matrix, 0)
    
    @staticmethod
    def get_largest(row):
        row_copy = row.copy()
        non_zero = row_copy[row_copy != 0]
        if len(non_zero) < 1:
            return None
        sorted_indices = np.argsort(row_copy)
        top_value = row_copy[sorted_indices[-1]]
        top_index = sorted_indices[-1]
        return top_value, top_index

    @classmethod
    def resolve_conflicts(cls, similarity_matrix, similarity_matrix_top):
        for _ in range(similarity_matrix.shape[0]):
            stop_flag = True
            for i in range(similarity_matrix.shape[0]):
                for j in range(similarity_matrix.shape[1]):
                    if similarity_matrix_top[i, j] == 0:
                        continue
                    for k in range(similarity_matrix.shape[0]):
                        if k == i or similarity_matrix_top[k, j] == 0:
                            continue
                        if similarity_matrix_top[k, j] >= similarity_matrix_top[i, j]:
                            similarity_matrix_top[i, j] = 0
                            similarity_matrix[i, j] = 0
                            second_largest = cls.get_largest(similarity_matrix[i])
                            if second_largest is not None:
                                second_index = second_largest[1]
                                similarity_matrix_top[i, second_index] = second_largest[0]
                            stop_flag = False
                            break
                        elif similarity_matrix_top[k, j] < similarity_matrix_top[i, j]:
                            similarity_matrix_top[k, j] = 0
                            similarity_matrix[k, j] = 0
                            second_largest = cls.get_largest(similarity_matrix[k])
                            if second_largest is not None:
                                second_index = second_largest[1]
                                similarity_matrix_top[k, second_index] = second_largest[0]
                            stop_flag = False
                            break
                    break
            if stop_flag:
                break
        return similarity_matrix_top
