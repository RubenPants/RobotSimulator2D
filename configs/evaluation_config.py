"""
evaluation_config.py

Configuration file corresponding the evaluation process of NEAT.
"""
from configs.base_config import BaseConfig
from utils.dictionary import *


class EvaluationConfig(BaseConfig):
    """Evaluation-specific configuration parameters."""
    
    __slots__ = {
        'fitness', 'fitness_comb', 'nn_k', 'safe_zone',
    }
    
    def __init__(self):
        # Fitness functions [distance, diversity, novelty, path]  TODO
        self.fitness: str = D_DISTANCE
        # Function to combine the fitness-values across different games, choices are: min, avg, max, gmean  [def=gmean]
        self.fitness_comb: str = D_GMEAN
        # Number of nearest neighbors taken into account for a NN-utilizing fitness function  [def=3]
        self.nn_k: int = 3
        # Safe zone during novelty search, neighbours outside this range are not taken into account  [def=1]
        self.safe_zone: float = 1
