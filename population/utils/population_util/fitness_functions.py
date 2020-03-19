"""
fitness_functions.py

This file contains multiple possible fitness functions. Each of the fitness functions takes in a dictionary with as key
 the ID of the corresponding candidate, and as value a list of all its final observations (i.e. list of game.close()
 dictionaries). Based on this input, a suitable fitness value for each of the candidates is determined.
"""
from math import sqrt

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from config import GameConfig
from utils.dictionary import *


# --------------------------------------------------> MAIN METHODS <-------------------------------------------------- #


def calc_pop_fitness(fitness_config: dict, game_observations: dict, game_params: list, generation: int):
    """
    Determine the fitness out of the given game_observation dictionary. This happens in two stages:
     1) Evaluate the candidate's fitness for each of the games individually, thus resolving in a list of floats
         (fitness) for each candidate for each of the games
     2) Combine all the fitness-values of all the games for every individual candidate to get the candidate's overall
         fitness score
    
    :param fitness_config: Configuration dictionary that contains:
        * D_FIT_COMB: Tag specifying in which way the fitness scores get combined (min, avg, max)
        * D_K: Number indicating on how many neighbors the k-nearest-neighbors algorithm is applied
        * D_TAG: Tag indicating which fitness-function is used
    :param game_observations: Dictionary containing for each genome the list of all its game.close() results
    :param game_params: List of game specific parameters for each of the used games (in order)
    :param generation: Current generation of the population
    :return: { genome_key: combined_fitness_float }
    """
    # 1) Evaluate fitness for each of the games
    intermediate_observations = fitness_per_game(
            fitness_config=fitness_config,
            game_observations=game_observations,
            game_params=game_params,
            generation=generation,
    )
    
    # 2) Combine the fitness-functions
    return fitness_averaged(fitness_config=fitness_config, fitness=intermediate_observations)


def fitness_averaged(fitness_config: dict, fitness: dict):
    """
    
    :param fitness_config: Configuration dictionary that contains a tag specifying in which way the fitness scores get
     combined (min, avg, max)
    :param fitness: { genome_key : [fitness_floats] }
    :return: Adjusted fitness dictionary: { genome_key: combined_fitness_float }
    """
    t = fitness_config[D_FIT_COMB]
    assert (t in [D_MIN, D_AVG, D_MAX, D_GMEAN])
    f = min if t == D_MIN else max if t == D_MAX else np.mean if D_AVG else stats.gmean
    for k in fitness.keys(): fitness[k] = f(fitness[k])
    return fitness


def fitness_per_game(fitness_config: dict, game_observations: dict, game_params: list, generation: int):
    """
    General fitness-function called by the evaluator environment containing all the possible attributes. Based on the
     given 'tag', determined by the population's config, a suitable fitness function is called.
    
    :param fitness_config: Configuration dictionary that contains:
        * D_K: Number indicating on how many neighbors the k-nearest-neighbors algorithm is applied
        * D_TAG: Tag indicating which fitness-function is used
    :param game_observations: Dictionary containing for each genome the list of all its game.close() results
    :param game_params: List of game specific parameters for each of the used games (in order)
    :param generation: Current generation of the population
    :return: Dictionary: { genome_key: [fitness_floats] }
    """
    tag = fitness_config[D_TAG]
    if tag == D_DISTANCE:
        return distance(
                game_observations=game_observations,
        )
    elif tag == D_NOVELTY:
        return novelty_search(
                game_observations=game_observations,
                game_params=game_params,
                k=fitness_config[D_K],
        )
    elif tag == D_PATH:
        return fitness_path(
                game_observations=game_observations,
                game_params=game_params,
        )
    elif tag == D_DIVERSITY:
        return diversity(
                game_observations=game_observations,
                game_params=game_params,
                gen=generation,
                k=fitness_config[D_K]
        )
    else:
        raise Exception(f"{tag} is not supported")


# -------------------------------------------------> HELPER METHODS <------------------------------------------------- #


def distance(game_observations: dict):
    """
    Determine the fitness based on the average distance to target in crows flight. This remaining distance is normalized
     by the A* distance from the agent's initial position. The fitness ranges between 0 and 1.
    
    TODO: Inspired by James' paper page --> Still need to implement it though!
    
    :param game_observations: Dictionary containing for each genome the list of all its game.close() results
    :return: { genome_id, [fitness_floats] }
    """
    cfg = GameConfig()
    diagonal = sqrt(cfg.x_axis ** 2 + cfg.y_axis ** 2)
    
    def get_score(d, reached=False):
        """Get a score for the given distance."""
        return 1 if reached else clip((1 - (d - cfg.target_reached) / diagonal) ** 2)
    
    fitness = dict()
    for k, v in game_observations.items():  # Iterate over the candidates
        fitness[k] = [get_score(o[D_DIST_TO_TARGET], reached=o[D_DONE]) for o in v]
    return fitness


def diversity(game_observations: dict, game_params: list, gen: int, k: int):
    """
    Every 10 generations, filter out the most fit candidates based on their distance towards the target, otherwise
     enforce novelty.
    
    :param game_observations: Dictionary containing for each genome the list of all its game.close() results
    :param game_params: List of game specific parameters for each of the used games (in order)
    :param gen: Population's current generation
    :param k: The number of neighbours taken into account
    :return: { genome_id, [fitness_floats] }
    """
    if gen % 10 == 0:
        return distance(game_observations=game_observations)
    else:
        return novelty_search(game_observations=game_observations, game_params=game_params, k=k)


def novelty_search(game_observations: dict, game_params: list, k: int = 5):
    """
    This method is used to apply the novelty-search across different games. This is the method that must be called by
     the evaluator-environment. The fitness ranges between 0 and 1.
    
    :param game_observations: Dictionary containing for each genome the list of all its game.close() results
    :param game_params: List of game specific parameters for each of the used games (in order)
    :param k: The number of neighbours taken into account
    :return: { genome_id, [fitness_floats] }
    """
    # Get base attributes
    candidates = list(game_observations.keys())
    n_games = len(list(game_observations.values())[0])
    
    # Init cumulative dictionary
    cum_novelty = dict()
    for c in candidates: cum_novelty[c] = []
    
    # Add novelty-scores for each of the candidates for each of the games
    for i in range(n_games):
        # Get observation for given game
        obs = dict()
        for c in candidates: obs[c] = game_observations[c][i]
        ns = novelty_search_game(obs, game_params, k=k)
        
        # Append to cumulative novelty score
        for c in candidates:
            cum_novelty[c].append(ns[c])
    return cum_novelty


# TODO: Not sure if proper Novelty function (add one by one?)
def novelty_search_game(game_observations: dict, game_params: list, k: int = 5):
    """
    Candidates are given a fitness based on how novel their position is. For each candidate, the distance to its k
     nearest neighbours is determined, which are then summed up with each other. Novel candidates are the ones that are
     far away from other candidates, thus have a (relative) low summed distance the the k-nearest neighbours. The idea
     behind novelty search is that it simulates an efficient search across the whole search-space, whilst only
     evaluating the candidates on their phenotype (final position).
    
    Briefly said: the further away a candidate is from the other candidates (in crows fly), the 'fitter' it is.
    
    :param game_observations: Dictionary containing for each genome the list of all its game.close() results
    :param game_params: List of game specific parameters for each of the used games (in order)
    :param k: The number of neighbours taken into account
    :return: Dictionary: key=genome_id, val=fitness of one game as a float
    """
    ids = list(game_observations.keys())
    assert (k > 0)  # Check if algorithm considers at least one neighbor
    assert (len(ids) > k)  # Check if k is not greater than population-size
    
    # Shuffle the ids randomly
    ids = list(np.random.permutation(ids))
    
    # Get the positions of the k first positions
    positions = np.asarray([game_observations[ids[i]][D_POS] for i in range(k + 1)])
    
    # Determine distance of first k neighbours
    distance = dict()
    knn = NearestNeighbors(n_neighbors=(k + 1)).fit(positions)  # k+1 since a candidate includes itself
    knn_distances, _ = knn.kneighbors(positions)  # Array of distances to k+1 nearest neighbors
    for i in range(k + 1):
        distance[ids[i]] = sum(knn_distances[i])
    
    # Iteratively append rest of candidates and determine distance
    for i in range(k + 1, len(ids)):
        positions = np.concatenate([positions, [game_observations[ids[i]][D_POS]]])
        knn = NearestNeighbors(n_neighbors=(k + 1)).fit(positions)
        knn_distances, _ = knn.kneighbors(positions)
        distance[ids[i]] = sum(knn_distances[i])
    
    # Normalize the distance
    max_distance = max(distance.values())
    for k in distance.keys(): distance[k] /= max_distance
    
    # Return result
    return distance


def fitness_path(game_observations: dict, game_params: list):
    """
    This metric uses the game-specific "path" values. This value indicates the quality for each for the tiles,
     indicating how far away this current tile is from target. Note that these values are normalized and transformed to
     fitness values, where the tile of the target has value 1 and the value of the tile with the longest path towards
     the target has value 0. The fitness ranges between 0 and 1.
    
    :param game_observations: Dictionary containing for each genome the list of all its game.close() results
    :param game_params: List of game specific parameters for each of the used games (in order)
    :return: { genome_id, [fitness_floats] }
    """
    paths = [g[D_PATH] for g in game_params]
    a_stars = [g[D_A_STAR] for g in game_params]
    
    def get_score(pos, g_index, reached=False):
        """Get a score for the given distance."""
        return 1 if reached else clip((1 - paths[g_index][round(pos[0], 1), round(pos[1], 1)] / a_stars[g_index]) ** 2)
    
    # Calculate the score
    fitness = dict()
    for k, v in game_observations.items():  # Iterate over the candidates
        fitness[k] = [get_score(o[D_POS], g_index=i, reached=o[D_DONE]) for i, o in enumerate(v)]
    return fitness


def clip(v):
    """Clip the value between 0 and 1."""
    return np.clip(v, a_min=0, a_max=1)
