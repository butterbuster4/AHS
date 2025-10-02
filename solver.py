import random
import time
from sklearn.cluster import SpectralClustering
import torch
from tile import Tile
import utils
from tqdm import tqdm 
from torchvision import models
from individual import Individual
from image_analysis import ImageAnalysis
from selection import roulette_selection
from crossover import Crossover
from evaluation import MNA_evaluation, MPR_evaluation, MABS_evaluation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_FOLDER = './images'  # Folder containing test images
MODEL_PATH_HORI = './models/hori_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth'
MODEL_PATH_VRTI = './models/vrti_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth'
MODEL_PATH_PEF = './models/best_efficientnetb3.pth'
PUZZLE_NUMBER = 5  # Number of images to select for the puzzle
PUZZLE_PIECE_NUMBER = 9  # Number of pieces in the puzzle (3x3 grid)
TILE_PER_ROW = 3  # Number of tiles per row in the puzzle
TILE_EFFECTIVE_SIZE = 96  # Effective size of each tile
GAP = 4  # Gap between tiles
PUZZLE_IMAGE_PATH = './puzzle.png'  # Path to the puzzle image
DEFAULT_GENERATIONS: int = 1000
DEFAULT_POPULATION: int = 1000
ELITE_NUMBER: int = 5
TERMINATION_THRESHOLD = 10


def initialize_population(population_size, initial_individual: Individual, all_tiles_count):
    population = [initial_individual]
    swaps_per_individual = max(1, int(all_tiles_count * 0.1))

    # 平均每种类型产生 N 个
    structured_count = population_size // 4
    random_count = population_size // 4
    swap_count = population_size - structured_count - random_count - 1

    rand_choice = random.choice
    rand_shuffle = random.shuffle

    # 1. 扰动型
    for _ in range(swap_count):
        groups = [list(g) for g in initial_individual.groups]  # 浅拷贝
        for _ in range(swaps_per_individual):
            source_group_list = [g for g in groups if len(g) > 0]
            if not source_group_list:
                break
            source_group = rand_choice(source_group_list)
            tile_to_move = rand_choice(source_group)
            source_group.remove(tile_to_move)
            target_group = rand_choice(groups)
            target_group.append(tile_to_move)
        population.append(Individual(groups))

    # 2. 完全随机型
    all_tiles = [tile for group in initial_individual.groups for tile in group]
    for _ in range(random_count):
        shuffled_tiles = random.sample(all_tiles, len(all_tiles))  # 一次性随机排列
        num_groups = len(initial_individual.groups)
        groups = [[] for _ in range(num_groups)]
        for i, tile in enumerate(shuffled_tiles):
            groups[i % num_groups].append(tile)
        population.append(Individual(groups))

    # 3. 组内随机型
    for _ in range(structured_count):
        groups = [list(g) for g in initial_individual.groups]  # 浅拷贝
        for group in groups:
            rand_shuffle(group)
        population.append(Individual(groups))

    return population


def get_elite_individuals(population, elites):
    """Returns first 'elite_count' fittest individuals from population"""
    return sorted(population, key=lambda x: x.fitness, reverse=True)[:elites]


def best_individual(population):
    """Returns the fittest individual from population"""
    return max(population, key=lambda x: x.fitness)


def start_evolution(tiles=None, puzzle_number=PUZZLE_NUMBER, crossover_rate=0.9):
    ImageAnalysis.analyze_puzzle(PUZZLE_IMAGE_PATH, tiles, DEVICE, MODEL_PATH_HORI, MODEL_PATH_VRTI)
    fittest = None
    best_fitness_score = float("-inf")
    termination_counter = 0
    
    # load perfection model
    pef_model = models.efficientnet_b3(pretrained=False)
    in_features = pef_model.classifier[1].in_features
    pef_model.classifier[1] = torch.nn.Linear(in_features, 2)
    pef_model.load_state_dict(torch.load(MODEL_PATH_PEF, map_location="cpu"))
    pef_model.to(DEVICE)

    # Cluster the tiles using spectral clustering to get initial groups
    spectral_model = SpectralClustering(
        n_clusters=puzzle_number,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=0
    )
    clusters = spectral_model.fit_predict(ImageAnalysis.max_similarity_matrix)
    tiles_groups = utils.divide_puzzle_into_groups_according_to_clusters(
        PUZZLE_IMAGE_PATH, ImageAnalysis.tiles, clusters
    )
    initial_individual = Individual(tiles_groups)

    # Create an individual from the groups
    population = initialize_population(DEFAULT_POPULATION, initial_individual, len(ImageAnalysis.tiles))

    # evolution loop
    for generation in tqdm(range(DEFAULT_GENERATIONS), desc="Evolving"):
        new_population = []

        # Elitism
        elite = get_elite_individuals(population, ELITE_NUMBER)
        new_population.extend(elite)

        selected_parents = roulette_selection(
            population, elites=ELITE_NUMBER
        )

        for first_parent, second_parent in selected_parents:
            if random.random() < crossover_rate:
                # 执行 crossover
                crossover = Crossover(first_parent, second_parent, pef_model)
                crossover.run()
                child = crossover.child()
                new_population.append(child)
            else:
                # 直接保留父代（随机一个）
                new_population.append(random.choice([first_parent, second_parent]))

        # 维护最优个体（避免每代全局扫描）
        gen_best = max(new_population, key=lambda x: x.fitness)
        print(f'Generation {generation}: Best Fitness = {gen_best.fitness}')
        if gen_best.fitness > best_fitness_score:
            fittest = gen_best
            best_fitness_score = gen_best.fitness
            termination_counter = 0
        else:
            termination_counter += 1

        if termination_counter == TERMINATION_THRESHOLD:
            fittest.to_image().save(f'best_individual_{fittest.fitness}.jpg')
            return fittest

        population = new_population
    return fittest


def solve():
    true_solution = utils.create_puzzle(IMAGE_FOLDER, PUZZLE_IMAGE_PATH, PUZZLE_NUMBER, TILE_PER_ROW, TILE_EFFECTIVE_SIZE, GAP)
    true_matrix = [[[None, None, None] for _ in range(3)] for _ in range(PUZZLE_NUMBER)]
    tiles = []
    for i, tile in enumerate(true_solution):
        image_index, row, col, image = tile
        tiles.append((image_index, row, col, Tile(image, i)))

    for i, tile in enumerate(tiles):
        image_index, row, col, true_tile = tile
        true_matrix[image_index][row][col] = true_tile

    true_individual = Individual(matrixs=true_matrix)

    result = start_evolution()
    result.to_image().save(f'best_individual_{result.fitness}.jpg')

    true_individual.to_image().save(f'true_individual_{true_individual.fitness}.jpg')

    utils.highlight_incorrect_tile_borders(result, true_individual, save_path="highlighted_borders_result.jpg")

    mna_score = MNA_evaluation(result.matrixs, true_individual.matrixs)
    mpr_score = MPR_evaluation(result.matrixs, true_individual.matrixs)
    mabs_score = MABS_evaluation(result.matrixs, true_individual.matrixs)
    print(f'MPR Score: {mpr_score}')
    print(f'MABS Score: {mabs_score}')
    print(f'MNA Score: {mna_score}')


def solve_with_tiles(tiles, puzzle_number, true_matrices):
    result = start_evolution(tiles, puzzle_number)
    true_individual = Individual(matrixs=true_matrices)
    utils.highlight_incorrect_tile_borders(result, true_individual, save_path="highlighted_borders_result.jpg")
    return result


if __name__ == "__main__":
    time_start = time.time()
    solve()
    time_end = time.time()
    print('totally cost', time_end - time_start)
