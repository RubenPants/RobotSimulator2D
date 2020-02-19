"""
population.py

This class is responsible for containing the population. It will simply be used as a container for the population's core
functionality such as its configuration and methods used to persist the population properly.
"""
import pickle
import re
from glob import glob

import matplotlib.pyplot as plt
import neat
from neat.math_util import mean

from configs.config import NeatConfig
from population.utils.genome import DefaultGenome
from population.utils.population_config import PopulationConfig
from population.utils.reporting import ReporterSet, StdOutReporter
from population.utils.statistics import StatisticsReporter
from population.visualizer import draw_net
from utils.dictionary import D_FIT_COMB, D_GAME_ID, D_K, D_POS, D_TAG
from utils.myutils import get_subfolder, update_dict


class CompleteExtinctionException(Exception):
    pass


class Population:
    def __init__(self,
                 name: str = "",
                 version: int = 0,
                 make_net_method=None,
                 query_net_method=None):
        """
        The population will be concerned about everything directly and solely related to the population. These are
        reporters, persisting methods, evolution and more.
        
        :param name: Name of population, used when specific population must be summon
        :param version: Version of the population, 0 if not versioned
        :param make_net_method: Method used to create a network based on a given genome
        :param query_net_method: Method used to query the action on the network, given a current state
        """
        # Set formal properties of the population
        if name:
            self.name = name
        else:
            cfg = NeatConfig()
            self.name = f"{cfg.fitness}{f'_{version}' if version else ''}"
        
        # Placeholders
        self.best_genome = None
        self.config = None
        self.fitness_config = None
        self.fitness_criterion = None
        self.generation = 0
        self.make_net = None
        self.population = None
        self.query_net = None
        self.reporters = None
        self.reproduction = None
        self.species = None
        
        # Try to load the population, create new if not possible
        if not self.load():
            assert (make_net_method is not None) and (query_net_method is not None)  # net-methods must be provided
            self.create_population(make_net_method=make_net_method, query_net_method=query_net_method)
    
    def __str__(self):
        return self.name
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def create_population(self, make_net_method, query_net_method):
        """
        Create a new population based on the given config file.
        
        :param make_net_method: Method used to create the genome-specific network
        :param query_net_method: Method used to query actions of the genome-specific network
        """
        # Init the population's configuration
        cfg = NeatConfig()
        config = PopulationConfig(
                genome_type=DefaultGenome,
                reproduction_type=neat.DefaultReproduction,
                species_set_type=neat.DefaultSpeciesSet,
                stagnation_type=neat.DefaultStagnation,
                config=cfg,
        )
        self.reporters = ReporterSet()
        self.set_config(config)
        
        # Fitness evaluation
        if self.config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif self.config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif self.config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not self.config.no_fitness_termination:
            raise RuntimeError(f"Unexpected fitness_criterion: {self.config.fitness_criterion!r}")
        
        # Config specific for fitness
        self.fitness_config = {
            D_FIT_COMB: cfg.fitness_comb,
            D_K:        cfg.nn_k,
            D_TAG:      cfg.fitness,
        }
        
        # Create a population from scratch, then partition into species
        print(self.config.genome_config)
        self.population = self.reproduction.create_new(self.config.genome_type,
                                                       self.config.genome_config,
                                                       self.config.pop_size)
        self.species = self.config.species_set_type(self.config.species_set_config, self.reporters)
        self.species.speciate(self.config, self.population, self.generation)
        
        # Create network method containers
        self.make_net = make_net_method
        self.query_net = query_net_method
        
        # The StatisticsReporter gathers and provides the most-fit genomes
        stats = StatisticsReporter()
        self.add_reporter(stats)
        
        # Use 'print' to output information about the run
        reporter = StdOutReporter()
        self.add_reporter(reporter)
        
        # Save newly made population
        self.save()
    
    def evolve(self):
        """
        The evolution-process consists out of two phases:
          1) Reproduction
          2) Speciation
        
        This method manipulates the population itself, so nothing has to be returned
        """
        # Create the next generation from the current generation
        self.population = self.reproduction.reproduce(self.config, self.species, self.config.pop_size, self.generation)
        
        # Check for complete extinction
        if not self.species.species:
            self.reporters.complete_extinction()
            
            # If requested by the user, create a completely new population, otherwise raise an exception
            if self.config.reset_on_extinction:
                self.population = self.reproduction.create_new(self.config.genome_type,
                                                               self.config.genome_config,
                                                               self.config.pop_size)
            else:
                raise CompleteExtinctionException()
        
        # Divide the new population into species
        self.species.speciate(self.config, self.population, self.generation)
        
        # Increment generation count
        self.generation += 1
    
    def create_blueprints(self, final_observations: dict, games: list):
        """
        Save images in the relative 'images/' subfolder of the population.
        
        :param final_observations: Dictionary of all the final game observations made
        :param games: List Game-objects used during evaluation
        """
        save_path = get_subfolder(f'population/storage/NEAT/{self}/', 'images')
        genome_keys = list(final_observations.keys())
        for g in games:
            # Get the game's blueprint
            g.get_blueprint()
            
            # Get all the final positions of the agents
            positions = []
            for gk in genome_keys:
                positions += [fo[D_POS] for fo in final_observations[gk] if fo[D_GAME_ID] == g.id]
            
            # Plot the positions
            dot_x = [p[0] for p in positions]
            dot_y = [p[1] for p in positions]
            plt.plot(dot_x, dot_y, 'ro')
            
            # Add target again to map
            plt.plot(0.5, g.y_axis - 0.5, 'go')
            
            # Add title
            plt.title("Blueprint - Game {id:05d} - Generation {gen:05d}".format(id=g.id, gen=self.generation))
            
            # Save figure
            game_path = get_subfolder(save_path, 'game{id:05d}'.format(id=g.id))
            plt.savefig('{gp}gen{gen:05d}'.format(gp=game_path, gen=self.generation))
            plt.close()
    
    def visualize_genome(self, genome=None, name: str = '', show: bool = True):
        """
        Visualize the architecture of the given genome.
        
        :param genome: Genome that must be visualized, best genome is chosen if none
        :param name: Name of the image, excluding the population's generation (auto concatenated)
        :param show: Directly visualize the architecture
        """
        if not genome:
            genome = self.best_genome
            if not name:
                name = 'best_genome_'
        name += 'gen_{gen:05d}'.format(gen=self.generation)
        get_subfolder(f'population/storage/NEAT/{self}/', 'images')
        sf = get_subfolder(f'population/storage/NEAT/{self}/images/', 'architectures')
        draw_net(self.config,
                 genome,
                 filename='{sf}{name}'.format(sf=sf, name=name),
                 view=show)
    
    # ---------------------------------------------> FUNCTIONAL METHODS <--------------------------------------------- #
    
    def add_evaluation_result(self, eval_result):
        """
        Append the result of the evaluation.
        
        :param eval_result: Dictionary
        """
        sf = get_subfolder(f'population/storage/NEAT/{self}/', 'evaluation')
        sf = get_subfolder(sf, f"{self.generation:05d}")
        update_dict(f'{sf}results', eval_result)
    
    def add_reporter(self, reporter):
        """
        Add a new reporter to the population's ReporterSet.
        """
        self.reporters.add(reporter)
    
    def remove_reporter(self, reporter):
        """
        Remove the given reporter from the population's ReporterSet.
        """
        self.reporters.remove(reporter)
    
    def set_config(self, cfg):
        """
        Update the config-file.
        """
        self.config = cfg
        stagnation = cfg.stagnation_type(cfg.stagnation_config, self.reporters)
        self.reproduction = cfg.reproduction_type(cfg.reproduction_config, self.reporters, stagnation)
    
    # ---------------------------------------------> PERSISTING METHODS <--------------------------------------------- #
    
    def save(self):
        """
        Save the population as the current generation.
        """
        # Create needed subfolder if not yet exist
        get_subfolder('population/storage/', 'NEAT')
        get_subfolder('population/storage/NEAT/', f'{self}')
        get_subfolder(f'population/storage/NEAT/{self}/', 'generations')
        
        # Save the population
        pickle.dump(self, open(f'population/storage/NEAT/{self}/generations/gen_{self.generation:05d}', 'wb'))
        print(f"Population '{self}' saved! Current generation: {self.generation}")
    
    def load(self, gen=None):
        """
        Load in a game, specified by its current id.
        
        :return: True: game successfully loaded | False: otherwise
        """
        try:
            if gen is None:
                # Load in all previous populations
                populations = glob(f'population/storage/NEAT/{self}/generations/gen_*')
                if not populations: raise FileNotFoundError
                
                # Find newest population and save generation number under 'gen'
                populations = [p.replace('\\', '/') for p in populations]
                regex = r"(?<=" + re.escape(f'population/storage/NEAT/{self}/generations/gen_') + ")[0-9]*"
                gen = max([int(re.findall(regex, p)[0]) for p in populations])
            
            # Load in the population under the specified generation
            pop = pickle.load(open(f'population/storage/NEAT/{self}/generations/gen_{gen:05d}', 'rb'))
            self.best_genome = pop.best_genome
            self.config = pop.config
            self.fitness_config = pop.fitness_config
            self.fitness_criterion = pop.fitness_criterion
            self.generation = pop.generation
            self.make_net = pop.make_net
            self.population = pop.population
            self.query_net = pop.query_net
            self.reporters = pop.reporters
            self.reproduction = pop.reproduction
            self.species = pop.species
            print(f"Population '{self}' loaded successfully! Current generation: {self.generation}")
            return True
        except FileNotFoundError:
            return False
