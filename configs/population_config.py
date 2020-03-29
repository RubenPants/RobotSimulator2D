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
        'genome_elitism', 'min_specie_size', 'parent_selection', 'pop_size', 'specie_count_max', 'specie_count_min',
        'specie_count_target', 'specie_elitism', 'specie_stagnation',
    }
    
    def __init__(self):
        # Individuals whose genetic distance is less than this threshold are in the same specie  [def=4.0]  TODO
        self.compatibility_thr: float = 4.
        # Sexual reproduction  [def=True]
        self.crossover_enabled: bool = False
        # Probability of having a crossover when crossover is enabled  [def=0.6]
        self.crossover_prob: float = 0.6
        # Number of generations before a previous elite specie can become stagnant  [def=5]
        self.elite_specie_stagnation: int = 5
        # The function used to compute the species fitness  [def=D_MAX]
        self.fitness_func: str = D_MAX
        # Number of most fit individuals per specie that are preserved as-is from one generation to the next  [def=2]
        self.genome_elitism: int = 1
        # Minimum number of genomes per species, keeping low prevents number of individuals blowing up  [def=16]  TODO
        self.min_specie_size: int = 16
        # The fraction for each species allowed to reproduce each generation (parent selection)  [def=0.2]  TODO
        self.parent_selection: float = 0.2
        # Maximum number of species  [def=15]
        self.specie_count_max = 15
        # Minimum number of species  [def=5]
        self.specie_count_min = 5
        # Target number of species  [def=10]
        self.specie_count_target = 10
        # Number of the best species that will be protected from stagnation  [def=2]
        self.specie_elitism: int = 2
        # Remove a specie if it hasn't improved over this many number of generations  [def=15]  TODO
        self.specie_stagnation: int = 15
        
        # TODO: Often used, hence placed outside of parameter-crowd
        # Number of individuals in each generation  [def=256]
        self.pop_size: int = 256
    
    def get_compatibility_threshold(self, n_species: int):
        """Get the compatibility threshold based on the current number of species."""
        # Lower the compatibility threshold when less than minimal number of species
        if n_species < self.specie_count_min:
            return self.compatibility_thr * 2 / 3
        
        # Return the default threshold when number of species is lower than or equal to the target number
        if n_species <= self.specie_count_target:
            return self.compatibility_thr
        
        # Linear scale between thr and thr*2 based on target and max specie count
        if n_species < self.specie_count_max:
            f = 1 + (n_species - self.specie_count_target) / (self.specie_count_max - self.specie_count_target)
            assert 1.0 <= f <= 2.0
            return self.compatibility_thr * f
        
        # Assign all the genomes to the closest species ones the maximum specie count is met
        return float("inf")
