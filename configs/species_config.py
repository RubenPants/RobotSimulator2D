"""
species_config.py

Configuration-file relating to the creation and manipulation of species.
"""
from configs.base_config import BaseConfig
from utils.dictionary import D_MAX


class SpeciesConfig(BaseConfig):
    """Specie-specific configuration parameters."""
    
    __slots__ = {
        'compatibility_threshold', 'elite_stagnation', 'elitism', 'fitness_func', 'max_number', 'stagnation'
    }
    
    def __init__(self):
        # Individuals whose genetic distance is less than this threshold are in the same specie  [def=3.0]  TODO
        self.compatibility_threshold: float = 2.0
        # Number of generations before a previous elite specie can become stagnant  [def=5]
        self.elite_stagnation: int = 5
        # Number of the best species that will be protected from stagnation  [def=2]  TODO
        self.elitism: int = 2
        # The function used to compute the species fitness  [def=D_MAX]
        self.fitness_func: str = D_MAX
        # Maximum number of species that can live along each other  [def=10]
        self.max_number: int = 128
        # Remove a specie if it hasn't improved over this many number of generations  [def=15]
        self.stagnation: int = 15
