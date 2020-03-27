"""
reproduction.py

Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.
"""
from __future__ import division

import copy
from itertools import count
from random import choice, random

from neat.math_util import mean
from neat.six_util import iteritems, itervalues

from config import Config
from population.utils.cache.genome_distance import GenomeDistanceCache
from population.utils.genome_util.genome import Genome
from population.utils.population_util.species import DefaultSpecies
from population.utils.population_util.stagnation import DefaultStagnation
from population.utils.reporter_util.reporting import ReporterSet


class DefaultReproduction:
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """
    
    def __init__(self, reporters: ReporterSet, stagnation: DefaultStagnation):
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = dict()
        self.previous_elites = set()
    
    def create_new(self, config: Config, num_genomes):
        """Create a new (random initialized) population."""
        new_genomes = dict()
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = Genome(key, num_outputs=config.genome.num_outputs, bot_config=config.bot)
            g.configure_new(config.genome)
            new_genomes[key] = g
            self.ancestors[key] = tuple()
        return new_genomes
    
    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)
        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            # Determine number of candidates in the specie, which is always at least the minimum specie-size
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size
            
            # Adjust the number of candidates in the population via a weighted average over a specie's previous size
            #   s is the number of candidates the specie will contain
            #   ps is the specie his previous size
            # Example: ps=64, s=32, new specie size will then be 48 (=64-16)
            spawn_amounts.append(ps + round((s - ps) / 2))
        
        # Normalize the spawn amounts so that the next generation is roughly the population size requested by the user
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]
        return spawn_amounts
    
    def reproduce(self, config: Config, species: DefaultSpecies, generation: int, logger=None):
        """Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents."""
        # Check which species to keep and which to remove
        remaining_fitness = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(config=config, species_set=species, gen=generation):
            # If specie is stagnant, then remove
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s, logger=logger)
            
            # Add the specie to the remaining species and save its (average adjusted) fitness
            else:
                remaining_fitness.extend(m.fitness for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)
        
        # If no species is left, force hard-reset
        if not remaining_species:
            species.species = dict()
            return dict()
        
        # Find minimum/maximum fitness across the entire population, for use in species adjusted fitness computation
        min_fitness = min(remaining_fitness)
        max_fitness = max(remaining_fitness)
        
        # Do not allow the fitness range to be zero, as we divide by it below
        fitness_range = max(0.1, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness, which is the normalized mean specie fitness (msf) divided by the number of
            #  candidates present in this specie
            msf = mean([m.fitness for m in itervalues(afs.members)])
            afs.adjusted_fitness = min((msf - min_fitness) / fitness_range, 1)
        
        # Minimum specie-size is defined by the number of elites and the minimal number of genomes in a population
        spawn_amounts = self.compute_spawn(adjusted_fitness=[s.adjusted_fitness for s in remaining_species],
                                           previous_sizes=[len(s.members) for s in remaining_species],
                                           pop_size=config.population.pop_size,
                                           min_species_size=max(config.population.min_specie_size,
                                                                config.population.genome_elitism))
        
        # Setup the next generation by filling in the new species with their elite, parents, and offspring
        new_population = dict()
        species.species = dict()
        for spawn_amount, specie in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species will always at least gets to retain its elites
            spawn_amount = max(spawn_amount, config.population.genome_elitism)
            
            # The species has at least one member for the next generation, so retain it
            assert spawn_amount > 0
            
            # Get all the specie's old (evaluated) members
            old_members = list(iteritems(specie.members))
            specie.members = dict()
            species.species[specie.key] = specie
            
            # Sort members in order of descending fitness (i.e. most fit members in front)
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)
            
            # Make sure that all a specie's elites are in the specie itself
            if config.population.genome_elitism > 0:
                # Add the current generation's elites to the population
                new_elites = set()
                for i, m in old_members[:config.population.genome_elitism]:
                    new_population[i] = m
                    new_elites.add((i, m))
                    spawn_amount -= 1
                
                # Add the previous generation's elites to the population (if not yet added and enough space)
                for i, m in self.previous_elites:
                    if i not in new_population and spawn_amount > 0:
                        new_population[i] = m
                        spawn_amount -= 1
                
                # Set current elite as previous_elites
                self.previous_elites = new_elites.copy()
            
            # If species is already completely full with its elite (not recommended), then go to next specie
            if spawn_amount <= 0: continue
            
            # Only use the survival threshold fraction to use as parents for the next generation, use at least all the
            #  elite of a population as parents
            reproduction_cutoff = max(round(config.population.parent_selection * len(old_members)),
                                      config.population.genome_elitism)
            
            # Use at least two parents no matter what the threshold fraction result is
            reproduction_cutoff = max(reproduction_cutoff, 2)
            parents = old_members[:reproduction_cutoff]
            
            # Fill the specie with offspring based on two randomly chosen parents
            while spawn_amount > 0:
                spawn_amount -= 1
                
                # Init genome dummy (values are overwritten later)
                gid = next(self.genome_indexer)
                child: Genome = Genome(gid, num_outputs=config.genome.num_outputs, bot_config=config.bot)
                
                # Choose the parents, note that if the parents are not distinct, crossover will produce a genetically
                #  identical clone of the parent (but with a different ID)
                p1_id, p1 = choice(parents)
                p2_id, p2 = choice(parents)
                if config.population.crossover_enabled and (p1_id != p2_id) and \
                        random() < config.population.crossover_prob:
                    child.configure_crossover(config=config.genome, genome1=p1, genome2=p2)
                else:
                    p2_id = p1_id
                    child.connections = copy.deepcopy(p1.connections)
                    child.nodes = copy.deepcopy(p1.nodes)
                
                # Keep mutating the child until it's valid
                valid = False
                while not valid:
                    child.mutate(config.genome)
                    valid = True
                    
                    # Check if the genome contains any connections
                    if len({c for c in child.connections.values() if c.enabled}) == 0:
                        valid = False
                        continue  # Continue the while-loop
                    
                    # Check if the genome is already in the population (i.e. did not mutate)
                    distances = GenomeDistanceCache(config=config.genome)
                    for genome in new_population.values():
                        if distances(genome0=child, genome1=genome) == 0:
                            valid = False
                            break  # Break the for-loop
                
                # Add the child to the population
                new_population[gid] = child
                self.ancestors[gid] = (p1_id, p2_id)
        
        return new_population
