"""
evaluation_config.py

Configuration file corresponding the evaluation process of NEAT.
"""
from configs.base_config import BaseConfig
from utils.dictionary import *


class EvaluationConfig(BaseConfig):
    """Evaluation-specific configuration parameters."""
    
    __slots__ = {
        'fitness', 'fitness_comb', 'fitness_criterion', 'fitness_threshold', 'nn_k', 'no_fitness_termination',
        'safe_zone'
    }
    
    def __init__(self):
        # Fitness functions [distance, diversity, novelty, path]  TODO
        self.fitness: str = D_DISTANCE
        # Function to combine the fitness-values across different games, choices are: min, avg, max, gmean  [def=gmean]
        self.fitness_comb: str = D_GMEAN
        # The function used to compute the termination criterion from the set of genome fitness [def=D_MAX]
        self.fitness_criterion: str = D_MAX
        # When fitness computed by fitness_criterion meets this threshold, the evolution process will terminate  [def=1]
        self.fitness_threshold: int = 1
        # Number of nearest neighbors taken into account for a NN-utilizing fitness function  [def=3]
        self.nn_k: int = 3
        # Don't consider fitness_criterion and fitness_threshold  [def=True]
        self.no_fitness_termination: bool = True
        # Safe zone during novelty search, neighbours outside this range are not taken into account  [def=1]
        self.safe_zone: float = 1
