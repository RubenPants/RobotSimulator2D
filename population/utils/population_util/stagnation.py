"""
stagnation.py

Keeps track of whether species are making progress and indicates the ones that are not to be removed.
"""
from neat.math_util import stat_functions
from neat.six_util import iteritems

from population.utils.config.default_config import ConfigParameter, DefaultClassConfig
from utils.dictionary import D_MAX


class DefaultStagnation(DefaultClassConfig):
    """Keeps track of whether species are making progress and helps remove ones that are not."""
    
    @classmethod
    def parse_config(cls, param_dict):  # TODO: Duplicate with species.py!
        return DefaultClassConfig(param_dict, [ConfigParameter('compatibility_threshold', float, 2.0),
                                               ConfigParameter('max_stagnation', int, 15),
                                               ConfigParameter('species_elitism', int, 2),
                                               ConfigParameter('species_fitness_func', str, D_MAX),
                                               ConfigParameter('species_max', int, 15),
                                               ConfigParameter('specie_stagnation', int, 5),
                                               ])
    
    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_config = config
        self.reporters = reporters
        self.specie_elites = dict()
        
        self.species_fitness_func = stat_functions.get(config.species_fitness_func)
        if self.species_fitness_func is None:
            raise RuntimeError(f"Unexpected species fitness func: {config.species_fitness_func!r}")
    
    def update(self, species_set, generation):
        """
        Update each specie's fitness history information, checks if it has improved the last max_stagnation generations,
        and returns list with stagnant species that need to be removed.
        """
        # Update each of the species' fitness-related statistics (i.e. fitness, fitness_history, and last_improved)
        species_data = []
        for specie_id, specie in iteritems(species_set.species):
            # Get previous fitness
            prev_fitness = max(specie.fitness_history) if specie.fitness_history else float("-inf")
            
            # Update specie's fitness stats
            specie.fitness = self.species_fitness_func(specie.get_fitnesses())
            specie.fitness_history.append(specie.fitness)
            specie.adjusted_fitness = None
            if specie.fitness > prev_fitness: specie.last_improved = generation
            species_data.append((specie_id, specie))
        
        # Sort the species in ascending fitness order
        species_data.sort(key=lambda x: x[1].fitness, reverse=True)
        
        # Update the specie_elites (first increase stagnation by one, then reset counter for current elites)
        for k in self.specie_elites.copy():
            self.specie_elites[k] += 1
            if self.specie_elites[k] > self.species_config.specie_stagnation: self.specie_elites.pop(k)
        for idx, (specie_id, _) in enumerate(species_data):
            if idx < self.species_config.species_elitism:
                self.specie_elites[specie_id] = 0
            else:
                break
        
        # Define if the population is stagnant or not
        result = []
        for specie_id, specie in species_data:
            is_stagnant = False
            
            # Check if the current specie belongs to one of the elite species over the last specie_stagnation
            #  generations. Elite species cannot become stagnant.
            if specie_id not in self.specie_elites:
                stagnant_time = generation - specie.last_improved
                is_stagnant = stagnant_time >= self.species_config.max_stagnation
            
            # Append to the result
            result.append((specie_id, specie, is_stagnant))
        
        return result
