"""
reproduction.py

Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.
"""
from __future__ import division

import copy
import math
import random
from itertools import count

from neat.math_util import mean
from neat.six_util import iteritems, itervalues

from population.utils.config.default_config import ConfigParameter, DefaultClassConfig
from population.utils.genome_util.genome import DefaultGenome


class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """
    
    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2),
                                   ConfigParameter('sexual_reproduction', bool, True),
                                   ])
    
    def __init__(self, config, reporters, stagnation):
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}
    
    def create_new(self, genome_type, genome_config, num_genomes, logger=None):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config, logger=logger)
            new_genomes[key] = g
            self.ancestors[key] = tuple()
        return new_genomes
    
    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)
        
        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size
            
            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1
            spawn_amounts.append(spawn)
        
        # Normalize the spawn amounts so that the next generation is roughly the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]
        
        return spawn_amounts
    
    def reproduce(self, config, species, pop_size, generation, logger=None):
        """
        Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.
        """
        # TODO: I don't like this modification of the species and stagnation objects, because it requires internal
        #  knowledge of the objects.
        
        # Filter out stagnated species, collect the set of non-stagnated species members, and compute their average
        # adjusted fitness. The average adjusted fitness scheme (normalized to the interval [0, 1]) allows the use of
        # negative fitness values without interfering with the shared fitness scheme.
        # Note: only fitness of non-stagnated species are determined.
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s, logger=logger)
            else:
                all_fitnesses.extend(m.fitness for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)
        
        # No species left.
        if not remaining_species:
            species.species = {}
            return {}
        
        # Find minimum/maximum fitness across the entire population, for use in species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        
        # Do not allow the fitness range to be zero, as we divide by it below.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness, which is the average fitness of a specie divided by the number of candidates
            # present in this specie.
            msf = mean([m.fitness for m in itervalues(afs.members)])
            afs.adjusted_fitness = (msf - min_fitness) / fitness_range
        
        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
        self.reporters.info(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}", logger=logger)
        
        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        
        # Isn't the effective min_species_size going to be max(min_species_size, self.reproduction_config.elitism)?
        # That would probably produce more accurate tracking of population sizes and relative fitnesses... doing.
        # TODO: document.
        min_species_size = max(min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes, pop_size, min_species_size)
        
        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)
            
            assert spawn > 0
            
            # The species has at least one member for the next generation, so retain it.
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s
            
            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)
            
            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1
            
            if spawn <= 0: continue
            
            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold * len(old_members)))
            
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]
            
            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1
                
                # Init genome dummy (values are overwritten later)
                gid = next(self.genome_indexer)
                child: DefaultGenome = config.genome_type(gid)
                
                # Choose the parents, note that if the parents are not distinct, crossover will produce a genetically
                # identical clone of the parent (but with a different ID). Note: crossover with the same parent would
                # result in a identical copy of this parent.
                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)
                if self.reproduction_config.sexual_reproduction and (parent1_id != parent2_id):
                    child.configure_crossover(config=config.genome_config, genome1=parent1, genome2=parent2)
                else:
                    parent2_id, parent2 = None, None
                    child.connections = copy.deepcopy(parent1.connections)
                    child.nodes = copy.deepcopy(parent1.nodes)
                child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)
        
        return new_population
