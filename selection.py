import random
import bisect

def roulette_selection(population, elites=4):
    fitness_values = [individual.fitness for individual in population]
    probability_intervals = [
        sum(fitness_values[: i + 1]) for i in range(len(fitness_values))
    ]
    
    def select_individual():
        """Selects random individual from population based on fitess value"""
        random_select = random.uniform(0, probability_intervals[-1])
        selected_index = bisect.bisect_left(probability_intervals, random_select)
        return population[selected_index]
    
    selected = []
    for i in range(len(population) - elites):
        first, second = select_individual(), select_individual()
        selected.append((first, second))
        
    return selected

import random

def tournament_selection(population, elites, k=3):
    def select_one_parent():
        tournament_contenders = random.sample(population, k)
        
        winner = max(tournament_contenders, key=lambda individual: individual.fitness)
        return winner

    selected_pairs = []
    num_offspring = len(population) - elites
    
    for _ in range(num_offspring):
        parent1 = select_one_parent()
        parent2 = select_one_parent()
        selected_pairs.append((parent1, parent2))
        
    return selected_pairs