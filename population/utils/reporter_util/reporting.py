"""
reporting.py

Makes possible reporter classes, which are triggered on particular events and may provide information to the user,
may do something else such as checkpointing, or may do both.
"""
from __future__ import division, print_function

import time

from neat.math_util import mean, stdev
from neat.six_util import iterkeys, itervalues


class ReporterSet(object):
    """
    Keeps track of the set of reporters
    and gives methods to dispatch them at appropriate points.
    """
    
    def __init__(self):
        self.reporters = []
    
    def add(self, reporter):
        self.reporters.append(reporter)
    
    def remove(self, reporter):
        self.reporters.remove(reporter)
    
    def start_generation(self, gen):
        for r in self.reporters: r.start_generation(gen)
    
    def end_generation(self, config, population, species_set):
        for r in self.reporters: r.end_generation(config, population, species_set)
    
    def post_evaluate(self, config, population, species, best_genome):
        for r in self.reporters: r.post_evaluate(config, population, species, best_genome)
    
    def post_reproduction(self, config, population, species):
        for r in self.reporters: r.post_reproduction(config, population, species)
    
    def complete_extinction(self):
        for r in self.reporters: r.complete_extinction()
    
    def found_solution(self, config, generation, best):
        for r in self.reporters: r.found_solution(config, generation, best)
    
    def species_stagnant(self, sid, species):
        for r in self.reporters: r.species_stagnant(sid, species)
    
    def info(self, msg):
        for r in self.reporters: r.info(msg)


class BaseReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""
    
    def start_generation(self, generation):
        pass
    
    def end_generation(self, config, population, species_set):
        pass
    
    def post_evaluate(self, config, population, species, best_genome):
        pass
    
    def post_reproduction(self, config, population, species):
        pass
    
    def complete_extinction(self):
        pass
    
    def found_solution(self, config, generation, best):
        pass
    
    def species_stagnant(self, sid, species):
        pass
    
    def info(self, msg):
        pass


class StdOutReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""
    
    def __init__(self):
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
    
    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()
    
    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
        sids = list(iterkeys(species_set.species))
        sids.sort()
        print("\t SID    age    size    fitness    adj fit    stag ")
        print("\t=====  =====  ======  =========  =========  ======")
        for sid in sids:
            s = species_set.species[sid]
            a = self.generation - s.created
            n = len(s.members)
            f = "--" if s.fitness is None else f"{s.fitness:.3f}"
            af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.3f}"
            st = self.generation - s.last_improved
            print(f"\t{sid:^5}  {a:^5}  {n:^6}  {f:^9}  {af:^9}  {st:^6}")
        
        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        if self.num_extinctions > 0:
            print(f'Total extinctions: {self.num_extinctions:d}')
        if len(self.generation_times) > 1:
            print(f"Generation time: {elapsed:.3f} sec  ({average:.3f} average)")
        else:
            print(f"Generation time: {elapsed:.3f} sec")
    
    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [c.fitness for c in itervalues(population)]
        # Full population
        print(f'Full population\'s fitness overview:')
        print(f'\t-       best fitness: {max(fitnesses):3.5f}')
        print(f'\t-       mean fitness: {mean(fitnesses):3.5f}')
        print(f'\t-      worst fitness: {min(fitnesses):3.5f}')
        print(f'\t- standard deviation: {stdev(fitnesses):3.5f}')
        # Best genome
        best_species_id = species.get_species_id(best_genome.key)
        print(f'Best genome overview:')
        print(f'\t- fitness: {best_genome.fitness:3.5f}')
        print(f'\t- size (hid, conn): {best_genome.size()!r}')
        print(f'\t- genome id: {best_genome.key}')
        print(f'\t- belongs to specie: {best_species_id}')
    
    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')
    
    def found_solution(self, config, generation, best):
        print(f'\nBest individual in generation {self.generation} meets fitness threshold - size: {best.size()!r}')
    
    def species_stagnant(self, sid, species):
        print(f"\nSpecies {sid} with {len(species.members)} members is stagnated: removing it")
    
    def info(self, msg):
        print(msg)
