"""
fitness_functions.py

This file contains multiple possible fitness functions. Each of the fitness functions takes in a dictionary with as key
the ID of the corresponding candidate, and as value a list of all its final observations (i.e. list of game.close()
dictionaries). Based on this input, a suitable fitness value for each of the candidates is determined.
"""
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from utils.dictionary import *


# --------------------------------------------------> MAIN METHODS <-------------------------------------------------- #

def calc_pop_fitness(fitness_config, game_observations):
    """
    Determine the fitness out of the given game_observation dictionary. This happens in two stages:
      1) Evaluate the candidate's fitness for each of the games individually, thus resolving in a list of floats
         (fitness) for each candidate for each of the games
      2) Combine all the fitness-values of all the games for every individual candidate to get the candidate's overall
         fitness score
    
    :param fitness_config: Configuration dictionary which contains:
        * D_FIT_COMB: Tag specifying in which way the fitness scores get combined (min, avg, max)
        * D_K: Number indicating on how many neighbors the k-nearest-neighbors algorithm is applied
        * D_TAG: Tag indicating which fitness-function is used
    :param game_observations: List of game.close() results (Dictionary)
    :return: { genome_key: combined_fitness_float }
    """
    # 1) Evaluate fitness for each of the games
    intermediate_observations = fitness_per_game(fitness_config=fitness_config, game_observations=game_observations)
    
    # 2) Combine the fitness-functions
    return fitness_averaged(fitness_config=fitness_config, fitness_dict=intermediate_observations)


def fitness_averaged(fitness_config: dict, fitness_dict: dict):
    """
    
    :param fitness_config: Configuration dictionary which contains:
        * D_FIT_COMB: Tag specifying in which way the fitness scores get combined (min, avg, max)
    :param fitness_dict: { genome_key : [fitness_floats] }
    :return: Adjusted fitness dictionary: { genome_key: combined_fitness_float }
    """
    t = fitness_config[D_FIT_COMB]
    assert (t in ['min', 'avg', 'max', 'gmean'])
    f = min if t == 'min' else max if t == 'max' else np.mean if 'avg' else stats.gmean
    for k in fitness_dict.keys():
        fitness_dict[k] = f(fitness_dict[k])
    return fitness_dict


def fitness_per_game(fitness_config: dict, game_observations):
    """
    General fitness-function called by the evaluator environment containing all the possible attributes. Based on the
    given 'tag', determined by the population's config, a suitable fitness function is called.
    
    :param fitness_config: Configuration dictionary which contains:
        * D_K: Number indicating on how many neighbors the k-nearest-neighbors algorithm is applied
        * D_TAG: Tag indicating which fitness-function is used
    :param game_observations: List of game.close() results (Dictionary)
    :return: Dictionary: { genome_key: [fitness_floats] }
    """
    # Check if valid tag is given
    tag = fitness_config[D_TAG]
    assert (tag in ['distance', 'distance_norm', 'novelty', 'quality_diversity'])
    
    if tag == 'distance':
        return fitness_distance(game_observations)
    elif tag == 'distance_norm':
        return fitness_distance_normalized(game_observations)
    elif tag == 'novelty':
        return novelty_search(game_observations=game_observations,
                              k=fitness_config[D_K])
    else:  # quality_diversity
        raise NotImplemented


# -------------------------------------------------> HELPER METHODS <------------------------------------------------- #


def fitness_distance(game_observations):
    """
    Determine the fitness based on the average distance to target in crows flight. This fitness is calculated as the
    inverted average distance, times one hundred.
    
    :param game_observations: List of game.close() results (Dictionary)
    :return: { genome_id, [fitness_floats] }
    """
    fitness_dict = dict()
    for k, v in game_observations.items():  # Iterate over the candidates
        fitness_dict[k] = [1 / o[D_DIST_TO_TARGET] for o in v]
    return fitness_dict


def fitness_distance_normalized(game_observations):
    """
    Determine the fitness based on the normalized average distance to the target (crows flight). The difference with the
    default 'fitness_distance' is that in this case, the distance is normalized over the diagonal of the maze (i.e.
    maximum distance). which is (roughly) 1000.
    
    :param game_observations: List of game.close() results (Dictionary)
    :return: Dictionary: key=genome_id, val=average fitness as a float
    """
    fitness_dict = dict()
    for k, v in game_observations.items():  # Iterate over the candidates
        fitness_dict[k] = [1 - o[D_DIST_TO_TARGET] / 1000 for o in v]
    return fitness_dict


def novelty_search(game_observations, k: int = 5):
    """
    This method is used to apply the novelty-search across different games. This is the method that must be called by
    the evaluator-environment.
    
    :param game_observations: List of game.close() results (Dictionary)
    :param k: The number of neighbours taken into account
    :return: Dictionary: key=genome_id, val=average fitness as a float
    """
    # Get base attributes
    candidates = list(game_observations.keys())
    n_games = len(list(game_observations.values())[0])
    
    # Init cumulative dictionary
    cum_novelty = dict()
    for c in candidates:
        cum_novelty[c] = []
    
    # Add novelty-scores for each of the candidates for each of the games
    for i in range(n_games):
        # Get observation for given game
        obs = dict()
        for c in candidates:
            obs[c] = game_observations[c][i]
        ns = novelty_search_game(obs, k=k)
        
        # Append to cumulative novelty score
        for c in candidates:
            cum_novelty[c].append(ns[c])
    return cum_novelty


def novelty_search_game(game_observations, k: int = 5):  # TODO: Not sure if properly Novelty Function (add one by one?)
    """
    Candidates are given a fitness based on how novel their position is. For each candidate, the distance to its k
    nearest neighbours is determined, which are then summed up with each other. Novel candidates are the ones that are
    far away from other candidates, thus have a (relative) low summed distance the the k-nearest neighbours. The idea
    behind novelty search is that it simulates an efficient search across the whole search-space, whilst only evaluating
    the candidates on their phenotype (final position).
    
    Briefly said: the further away a candidate is from the other candidates (in crows fly), the 'fitter' it is.
    
    :param game_observations: Dictionary of List of game.close() results (Dictionary)
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
    
    # Return result
    return distance
