"""
stagnation.py

Keeps track of whether species are making progress and indicates the ones that are not to be removed.
"""
from neat.math_util import stat_functions
from neat.six_util import iteritems

from config import Config
from configs.species_config import SpeciesConfig


class DefaultStagnation:
    """Keeps track of whether species are making progress and helps remove ones that are not."""
    
    def __init__(self, config: SpeciesConfig, reporters):
        # pylint: disable=super-init-not-called
        self.reporters = reporters
        self.specie_elites = dict()
        
        self.species_fitness_func = stat_functions.get(config.fitness_func)
        if self.species_fitness_func is None:
            raise RuntimeError(f"Unexpected species fitness func: {config.fitness_func!r}")
    
    def update(self, config: Config, species_set, gen):
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
            if specie.fitness > prev_fitness: specie.last_improved = gen
            species_data.append((specie_id, specie))
        
        # Sort the species in ascending fitness order
        species_data.sort(key=lambda x: x[1].fitness, reverse=True)
        
        # Update the specie_elites (first increase stagnation by one, then reset counter for current elites)
        for k in self.specie_elites.copy():
            self.specie_elites[k] += 1
            if self.specie_elites[k] > config.species.stagnation: self.specie_elites.pop(k)
        for idx, (specie_id, _) in enumerate(species_data):
            if idx < config.species.elitism:
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
                stagnant_time = gen - specie.last_improved
                is_stagnant = stagnant_time >= config.species.elite_stagnation
            
            # Append to the result
            result.append((specie_id, specie, is_stagnant))
        
        return result
