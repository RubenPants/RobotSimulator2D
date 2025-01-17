"""
species.py

Divides the population into species based on genomic distances.
"""
from itertools import count

from neat.math_util import mean, stdev
from neat.six_util import iteritems, iterkeys, itervalues

from config import Config
from population.utils.cache.genome_distance import GenomeDistanceCache


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
            
            # Sort the genomes based on their distance towards the specie
            sorted_repr = sorted(candidates, key=lambda x: x[0])
            
            # Go through closest genomes to find first with different architecture than other specie's representative
            added = False
            for _, new_repr in sorted_repr:
                match = [sid for sid, rid in new_representatives.items() if
                         distances.get_disjoint_genes(population[rid], new_repr) == (0, 0)]
                if len(match) == 0:
                    new_representatives[sid] = new_repr.key
                    members[sid] = [new_repr.key]
                    unspeciated.remove(new_repr.key)
                    added = True
                    break
            
            # If no new representative found that is unique, add the closest
            if not added:
                new_repr = sorted_repr[0][1]
                new_representatives[sid] = new_repr.key
                members[sid] = [new_repr.key]
                unspeciated.remove(new_repr.key)
        
        # Partition the population into species based on their genetic similarity
        while unspeciated:
            # Pull unspeciated genome
            gid = unspeciated.pop()
            genome = population[gid]
            
            # Check if a specie already exists with the same architecture, if so, add the genome to this specie
            match = [sid for sid, rid in new_representatives.items() if
                     distances.get_disjoint_genes(population[rid], genome) == (0, 0)]
            if match:
                # Append to closest specie
                if len(match) == 1:
                    members[match[0]].append(gid)
                    continue
                else:
                    temp = sorted([(distances(population[new_representatives[sid]], genome), sid) for sid in match],
                                  key=lambda x: x[0])  # Sort on minimal distance
                    members[temp[0][1]].append(gid)
                    continue
            
            # Find the species with the most similar representative
            specie_distance = []
            candidates = []
            for sid, rid in iteritems(new_representatives):
                rep = population[rid]
                d = distances(rep, genome)
                specie_distance.append((d, sid))
                if d < config.population.get_compatibility_threshold(n_species=len(new_representatives)):
                    candidates.append((d, sid))
            
            # There are species close enough; add genome to most similar specie
            if candidates:
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
                f'\n\t- Maximum: {max(itervalues(distances.distances)):.5f}'
                f'\n\t- Mean: {mean(itervalues(distances.distances)):.5f}'
                f'\n\t- Minimum: {min(itervalues(distances.distances)):.5f}'
                f'\n\t- Standard deviation: {stdev(itervalues(distances.distances)):.5f}',
                logger=logger,
                print_result=False,
        )
        print(f'Maximum genetic distance: {round(max(itervalues(distances.distances)), 3)}')  # Print reduced version
        
        # Report over the most complex genome
        most_complex = sorted([(g.size(), gid) for gid, g in population.items()], key=lambda x: x[0], reverse=True)[0]
        gid = most_complex[1]
        size = most_complex[0]
        specie_id = self.genome_to_species[gid]
        distance = distances(population[gid], self.species[specie_id].representative)
        self.reporters.info(
                f"Most complex genome '{gid}' "
                f"of size {size} "
                f"belongs to specie '{specie_id}' "
                f"with distance to representative of {round(distance, 3)}",
                logger=logger,
        )
    
    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]
    
    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
