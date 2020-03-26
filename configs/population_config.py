"""
reproduction_config.py

Configuration file corresponding the reproduction of a NEAT-population.
"""
from configs.base_config import BaseConfig
from utils.dictionary import D_MAX


class PopulationConfig(BaseConfig):
    """Reproduction-specific configuration parameters."""
    
    __slots__ = {
        'compatibility_thr', 'crossover_enabled', 'crossover_prob', 'elite_specie_stagnation', 'fitness_func',
        'genome_elitism', 'min_specie_size', 'parent_selection', 'pop_size', 'specie_elitism', 'specie_stagnation'
    }
    
    def __init__(self):
        # Individuals whose genetic distance is less than this threshold are in the same specie  [def=2.0]  TODO
        self.compatibility_thr: float = 2.0
        # Sexual reproduction  [def=True]
        self.crossover_enabled: bool = True
        # Probability of having a crossover when crossover is enabled  [def=0.6]
        self.crossover_prob: float = 0.6
        # Number of generations before a previous elite specie can become stagnant  [def=5]
        self.elite_specie_stagnation: int = 5
        # The function used to compute the species fitness  [def=D_MAX]
        self.fitness_func: str = D_MAX
        # Number of most fit individuals per specie that are preserved as-is from one generation to the next  [def=3]
        self.genome_elitism: int = 3
        # Minimum number of genomes per species, keeping low prevents number of individuals blowing up  [def=10]  TODO
        self.min_specie_size: int = 5
        # The fraction for each species allowed to reproduce each generation (parent selection)  [def=0.3]  TODO
        self.parent_selection: float = 0.3
        # Number of the best species that will be protected from stagnation  [def=1]
        self.specie_elitism: int = 1
        # Remove a specie if it hasn't improved over this many number of generations  [def=15]
        self.specie_stagnation: int = 25
        
        # TODO: Often used, hence placed outside of crowd
        # Number of individuals in each generation  [def=128]  TODO
        self.pop_size: int = 32
