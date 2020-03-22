"""
species.py

Divides the population into species based on genomic distances.
"""
from itertools import count

from neat.math_util import mean, stdev
from neat.six_util import iteritems, iterkeys, itervalues

from population.utils.config.default_config import ConfigParameter, DefaultClassConfig
from utils.dictionary import D_MAX


class Species(object):
    """Create a specie, which is a container for similar genomes."""
    
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative = None
        self.members = dict()
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []
    
    def update(self, representative, members):
        self.representative = representative
        self.members = members
    
    def get_fitnesses(self):
        return [m.fitness for m in itervalues(self.members)]


class GenomeDistanceCache(object):
    """Makes sure that redundant distance-computations will not occur. (e.g. d(1,2)==d(2,1))."""
    
    def __init__(self, config):
        self.distances = dict()
        self.config = config
    
    def __call__(self, genome0, genome1):
        """Calculate the genetic distances between two genomes and store in cache if not yet present."""
        g0 = genome0.key
        g1 = genome1.key
        if g0 > g1: g0, g1 = g1, g0  # Lowest key is always first, removes redundant distance checks
        d = self.distances.get((g0, g1))  # Safe-get, returns result or None if key does not exist
        if d is None:
            d = genome0.distance(genome1, self.config)
            self.distances[g0, g1] = d
        return d


class DefaultSpecies(DefaultClassConfig):
    """ Encapsulates the default speciation scheme. """
    
    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = dict()
        self.genome_to_species = dict()
    
    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict, [ConfigParameter('compatibility_threshold', float, 2.0),
                                               ConfigParameter('max_stagnation', int, 15),
                                               ConfigParameter('species_elitism', int, 2),
                                               ConfigParameter('species_fitness_func', str, D_MAX),
                                               ConfigParameter('species_max', int, 15),
                                               ConfigParameter('specie_stagnation', int, 5),
                                               ])
    
    def speciate(self, config, population, generation, logger=None):
        """
        Place genomes into species by genetic similarity.

        :note: This method assumes the current representatives of the species are from the old generation, and that
         after speciation has been performed, the old representatives should be dropped and replaced with
         representatives from the new generation. If you violate this assumption, you should make sure other necessary
         parts of the code are updated to reflect the new behavior.
        """
        assert isinstance(population, dict)
        unspeciated = set(iterkeys(population))  # Initially the full population
        
        # Add each of the genomes that already belongs to a given specie to that specie
        members = dict()  # Dictionary containing for each specie its corresponding members
        for genome_id in unspeciated.copy():
            if genome_id in self.genome_to_species:
                specie_id = self.genome_to_species[genome_id]
                if specie_id not in members: members[specie_id] = set()
                members[specie_id].add(genome_id)
                unspeciated.remove(genome_id)
        
        # Find the best representatives for each existing species based on its remaining genomes (parents)
        distances = GenomeDistanceCache(config.genome_config)  # Keeps current distances between genomes
        new_representatives = dict()  # Updated representatives for each specie (least difference with previous)
        for specie_id, specie in iteritems(self.species):
            assert len(members[specie_id]) > 0  # Each specie must have its parents
            
            # Calculate the distances for each of the specie's parents relative to the specie's previous representative
            specie_distance = []
            for genome_id in members[specie_id]:
                genome = population[genome_id]
                distance = distances(specie.representative, genome)
                specie_distance.append((distance, genome))
            
            # Set as the new representative the genome closest to the current representative
            _, new_representative = min(specie_distance, key=lambda x: x[0])
            new_representatives[specie_id] = new_representative.key
        
        # Partition population into species based on genetic similarity
        while unspeciated:
            genome_id = unspeciated.pop()  # Pop a genome-identifier from the unspeciated-set
            genome = population[genome_id]  # Load in the genome corresponding the identifier
            
            # No species exist yet, create one
            if not new_representatives:
                specie_id = next(self.indexer)
                if specie_id not in members: members[specie_id] = set()
                new_representatives[specie_id] = genome_id
                members[specie_id].add(genome_id)
                continue
            
            # Determine the distance to each specie's representative
            specie_distance = []
            for specie_id, representative_id in iteritems(new_representatives):
                rep = population[representative_id]
                distance = distances(rep, genome)
                specie_distance.append((distance, specie_id))
            
            # Determine the smallest distance
            smallest_distance, closest_specie_id = min(specie_distance, key=lambda x: x[0])
            
            # Check if distance falls within threshold and maximum number of species is not exceeded
            if (smallest_distance < self.species_config.compatibility_threshold) or \
                    (len(new_representatives) >= self.species_config.species_max):
                members[closest_specie_id].add(genome_id)  # Add genome to closest specie
            else:  # Create new specie with genome exceeding threshold as representative
                specie_id = next(self.indexer)
                if specie_id not in members: members[specie_id] = set()
                new_representatives[specie_id] = genome_id
                members[specie_id].add(genome_id)
        
        # Update species collection based on new speciation
        self.genome_to_species = dict()  # Dictionary mapping genome to specie identifiers
        for specie_id, representative_id in iteritems(new_representatives):
            specie = self.species.get(specie_id)
            
            # Create specie if not yet existed
            if specie is None:
                specie = Species(specie_id, generation)
                self.species[specie_id] = specie
            
            # Update the specie's current members with all the new members
            specie_members = members[specie_id]
            for genome_id in specie_members: self.genome_to_species[genome_id] = specie_id
            
            # Update the specie's current representative and members
            member_dict = dict((genome_id, population[genome_id]) for genome_id in specie_members)
            specie.update(population[representative_id], member_dict)
        
        # Report over the current species (distances)
        self.reporters.info(
                f'Genetic distance:'
                f'\n\t- Maximum: {max(itervalues(distances.distances)):.3f}'
                f'\n\t- Mean: {mean(itervalues(distances.distances)):.3f}'
                f'\n\t- Minimum: {min(itervalues(distances.distances)):.3f}'
                f'\n\t- Standard deviation: {stdev(itervalues(distances.distances)):.3f}',
                logger=logger,
                print_result=False,
        )
        print(f'Maximum genetic distance: {max(itervalues(distances.distances))}')  # Print reduced version
    
    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]
    
    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
