"""
species.py

Divides the population into species based on genomic distances.
"""
from itertools import count

from neat.math_util import mean, stdev
from neat.six_util import iteritems, iterkeys, itervalues

from config import Config
from configs.genome_config import GenomeConfig


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
    
    def __init__(self, config: GenomeConfig):
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


class DefaultSpecies:
    """ Encapsulates the default speciation scheme. """
    
    def __init__(self, reporters):
        self.reporters = reporters
        self.indexer = count(1)
        self.species = dict()
        self.genome_to_species = dict()
    
    def speciate(self, config: Config, population, generation, logger=None):
        """
        Place genomes into species by genetic similarity.

        :note: This method assumes the current representatives of the species are from the old generation, and that
         after speciation has been performed, the old representatives should be dropped and replaced with
         representatives from the new generation.
        """
        assert isinstance(population, dict)
        unspeciated = set(iterkeys(population))  # Initially the full population
        distances = GenomeDistanceCache(config.genome)  # Cache memorizing distances between genomes
        new_representatives = dict()  # Updated representatives for each specie (least difference with previous)
        members = dict()  # Dictionary containing for each specie its corresponding members
        
        # Update the representatives for each specie as the genome closest to the previous representative
        for sid, s in iteritems(self.species):
            candidates = []
            for gid in unspeciated:
                genome = population[gid]
                d = distances(s.representative, genome)
                candidates.append((d, genome))
            
            # The new representative is the genome closest to the current representative
            _, new_repr = min(candidates, key=lambda x: x[0])
            new_representatives[sid] = new_repr.key
            members[sid] = [new_repr.key]
            unspeciated.remove(new_repr.key)
        
        # Partition the population into species based on their genetic similarity
        while unspeciated:
            # Pull unspeciated genome
            gid = unspeciated.pop()
            genome = population[gid]
            
            # Find the species with the most similar representative
            candidates = []
            for sid, rid in iteritems(new_representatives):
                rep = population[rid]
                d = distances(rep, genome)
                if d < config.species.compatibility_threshold:
                    candidates.append((d, sid))
            
            if candidates:  # There are species close enough; add genome to most similar specie
                _, sid = min(candidates, key=lambda x: x[0])
                members[sid].append(gid)
            else:  # No specie is similar enough, create new specie with this genome as its representative
                sid = next(self.indexer)
                new_representatives[sid] = gid
                members[sid] = [gid]
        
        # Update the species collection based on the newly defined speciation
        self.genome_to_species = dict()
        for sid, rid in iteritems(new_representatives):
            s = self.species.get(sid)
            
            # One of the newly defined species
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s
            
            # Append the newly added members to the genome_to_species mapping
            specie_members = members[sid]
            for gid in specie_members:
                self.genome_to_species[gid] = sid
            
            # Update the specie's current representative and members
            member_dict = dict((genome_id, population[genome_id]) for genome_id in specie_members)
            s.update(representative=population[rid], members=member_dict)
        
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
