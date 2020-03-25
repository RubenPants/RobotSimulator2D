"""
reproduction_config.py

Configuration file corresponding the reproduction of a NEAT-population.
"""
from configs.base_config import BaseConfig


class ReproductionConfig(BaseConfig):
    """Reproduction-specific configuration parameters."""
    
    __slots__ = {
        'elitism', 'min_species_size', 'parent_selection', 'pop_size', 'sexual', 'sexual_prob',
    }
    
    def __init__(self):
        # Number of most fit individuals per specie that are preserved as-is from one generation to the next  [def=3]
        self.elitism: int = 3
        # The fraction for each species allowed to reproduce each generation (parent selection)  [def=0.3]  TODO
        self.parent_selection: float = 0.3
        # Minimum number of genomes per species, keeping low prevents number of individuals blowing up  [def=10]  TODO
        self.min_species_size: int = 10
        # Sexual reproduction  [def=True]
        self.sexual: bool = True
        # Probability of having a crossover when sexual=True  [def=0.6]
        self.sexual_prob: float = 0.6
        
        # TODO: Often used, hence placed outside of crowd
        # Number of individuals in each generation  [def=128]  TODO
        self.pop_size: int = 10
