"""
fitness_functions.py

This file contains multiple possible fitness functions. Each of the fitness functions takes in a dictionary with as key
 the ID of the corresponding candidate, and as value a list of all its final observations (i.e. list of game.close()
 dictionaries). Based on this input, a suitable fitness value for each of the candidates is determined.
"""
import sys
from math import sqrt

from numpy import clip, mean
from scipy.stats import gmean

from config import GameConfig
from configs.evaluation_config import EvaluationConfig
from utils.dictionary import *

if sys.platform == 'linux':
    from utils.cy.intersection_cy import line_line_intersection_cy as intersection
    from utils.cy.line2d_cy import Line2dCy as Line
else:
    from utils.intersection import line_line_intersection as intersection
    from utils.line2d import Line2d as Line


# --------------------------------------------------> MAIN METHODS <-------------------------------------------------- #


def calc_pop_fitness(fitness_config: EvaluationConfig, game_observations: dict, game_params: list, generation: int):
    """
    Determine the fitness out of the given game_observation dictionary. This happens in two stages:
     1) Evaluate the candidate's fitness for each of the games individually, thus resolving in a list of floats
         (fitness) for each candidate for each of the games
     2) Combine all the fitness-values of all the games for every individual candidate to get the candidate's overall
         fitness score
    
    :param fitness_config: EvaluationConfig object
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


def fitness_averaged(fitness_config: EvaluationConfig, fitness: dict):
    """
    
    :param fitness_config: Configuration dictionary that contains a tag specifying in which way the fitness scores get
     combined (min, avg, max)
    :param fitness: { genome_key : [fitness_floats] }
    :return: Adjusted fitness dictionary: { genome_key: combined_fitness_float }
    """
    t = fitness_config.fitness_comb
    assert (t in [D_MIN, D_AVG, D_MAX, D_GMEAN])
    f = min if t == D_MIN else max if t == D_MAX else mean if D_AVG else gmean
    for k in fitness.keys(): fitness[k] = f(fitness[k])
    return fitness


def fitness_per_game(fitness_config: EvaluationConfig, game_observations: dict, game_params: list, generation: int):
    """
    General fitness-function called by the evaluator environment containing all the possible attributes. Based on the
     given 'tag', determined by the population's config, a suitable fitness function is called.
    
    :param fitness_config: EvaluationConfig object
    :param game_observations: Dictionary containing for each genome the list of all its game.close() results
    :param game_params: List of game specific parameters for each of the used games (in order)
    :param generation: Current generation of the population
    :return: Dictionary: { genome_key: [fitness_floats] }
    """
    tag = fitness_config.fitness
    if tag == D_DISTANCE:
        return distance(
                game_observations=game_observations,
        )
    elif tag == D_NOVELTY:
        return novelty_search(
                game_observations=game_observations,
                game_params=game_params,
                k=fitness_config.nn_k,
                safe_zone=fitness_config.safe_zone,
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
                k=fitness_config.nn_k,
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
        return 1 if reached else clip_f((1 - (d - cfg.target_reached) / diagonal) ** 2)
    
    fitness = dict()
    for k, v in game_observations.items():  # Iterate over the candidates
        fitness[k] = [get_score(o[D_DIST_TO_TARGET], reached=o[D_DONE]) for o in v]
    return fitness


def diversity(game_observations: dict, game_params: list, gen: int, k: int = 3, safe_zone: float = 1):
    """
    Every end of 10 generations, filter out the most fit candidates based on their distance towards the target,
     otherwise for enforce novelty.
    
    :param game_observations: Dictionary containing for each genome the list of all its game.close() results
    :param game_params: List of game specific parameters for each of the used games (in order)
    :param gen: Population's current generation
    :param k: The number of neighbours taken into account
    :param safe_zone: The range surrounding a genome in which other neighbours are taken into account
    :return: { genome_id, [fitness_floats] }
    """
    if (gen + 1) % 10 == 0:
        return distance(game_observations=game_observations)
    else:
        return novelty_search(game_observations=game_observations, game_params=game_params, k=k, safe_zone=safe_zone)


def novelty_search(game_observations: dict, game_params: list, k: int = 3, safe_zone: int = 1):
    """
    Rate a genome based on its novelty. A 'more novel' genomes is further placed away from its peers than another
     genome. This novelty is based on the final position of the genome. A genome gets a perfect score if no other
     genomes are within a 1 meter range of the genome's center or the genome reached the target.
     
    :note: Distance measures take walls into account. Two genomes close to each other with a wall between them are not
     considered nearby.
    
    :param game_observations: Dictionary containing for each genome (key) the list of all its game.close() results
    :param game_params: List of game specific parameters for each of the used games (in order)
    :param k: Number of closest neighbours taken into account
    :param safe_zone: The range surrounding a genome in which other neighbours are taken into account
    :return: { genome_id, [fitness_floats] }
    """
    # For each game, create a dictionary of the genome-id mapped to its position
    position_dict = dict()
    for game_id in range(len(game_params)): position_dict[game_id] = dict()
    for genome_id, observations in game_observations.items():
        for game_id, observation in enumerate(observations):
            position_dict[game_id][genome_id] = observation[D_POS]
    
    # Define the fitness for each genome at each game
    distance_dict = dict()
    for game_id, positions in position_dict.items():
        distance_dict[game_id] = dict()
        
        # Go over each genome to measure its lengths towards the other genomes
        cache = DistanceCache(safe_zone=safe_zone)
        for genome_id, genome_pos in positions.items():
            # Go over all the other genomes
            dist = set()
            for other_genome_id, other_genome_pos in positions.items():
                if genome_id == other_genome_id: continue
                d = cache.distance(pos1=genome_pos,
                                   pos2=other_genome_pos,
                                   walls=game_params[game_id][D_WALLS])
                if d < safe_zone: dist.add(d)
            
            # Add the k neighbours that are closest by
            distance_dict[game_id][genome_id] = sorted(dist)[:k]
    
    # Stitch results together such that each genome is mapped to a fitness-list
    fitness_dict = dict()
    for genome_id in game_observations.keys():
        fitness_dict[genome_id] = []
        for game_id in range(len(game_params)):
            score = (sum(distance_dict[game_id][genome_id]) / (safe_zone * k)) ** 2
            assert 0 <= score <= 1.0
            fitness_dict[genome_id].append(score)
    return fitness_dict


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
    
    def get_score(pos, gi, reached=False):
        """Get a score for the given path-position."""
        if reached:
            return 1
        temp = paths[gi][round(pos[0], 1), round(pos[1], 1)] / a_stars[gi]
        return (clip_f(1 - temp) + clip_f(1 - temp) ** 2) / 2
    
    # Calculate the score
    fitness = dict()
    for k, v in game_observations.items():  # Iterate over the candidates
        fitness[k] = [get_score(o[D_POS], gi=i, reached=o[D_DONE]) for i, o in enumerate(v)]
    return fitness


def clip_f(v):
    """Clip the value between 0 and 1."""
    return clip(v, a_min=0, a_max=1)


class DistanceCache:
    """Cache for the distance-checks."""
    
    def __init__(self, safe_zone):
        self.distances = dict()
        self.range = safe_zone
    
    def distance(self, pos1, pos2, walls):
        """Determine the distance between two positions. If the"""
        # Check cache
        if pos2 < pos1: pos2, pos1 = pos1, pos2
        l = Line(pos1, pos2)
        if l in self.distances: return self.distances[l]
        
        # Not in cache, determine distance
        intersect = False
        for wall in walls:
            inter, _ = intersection(l, wall)
            if inter:
                intersect = True
                break
        
        # Add the distance if the other genome is in the 1m-zone and no wall in between
        self.distances[l] = l.get_length() if not intersect and (l.get_length() < self.range) else float('inf')
        return self.distances[l]
