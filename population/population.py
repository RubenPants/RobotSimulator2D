"""
population.py

This class is responsible for containing the population. It will simply be used as a container for the population's core
functionality such as its configuration and methods used to persist the population properly.
"""
import re
from glob import glob

import matplotlib.pyplot as plt
from neat.math_util import mean

from configs.config import NeatConfig
from population.utils.config.default_config import Config
from population.utils.genome_util.genome import DefaultGenome
from population.utils.genome_util.genome_visualizer import draw_net
from population.utils.network_util.feed_forward_net import make_net
from population.utils.population_util.reproduction import DefaultReproduction
from population.utils.population_util.species import DefaultSpeciesSet
from population.utils.population_util.stagnation import DefaultStagnation
from population.utils.reporter_util.reporting import ReporterSet, StdOutReporter
from population.utils.reporter_util.statistics import StatisticsReporter
from utils.dictionary import D_FIT_COMB, D_GAME_ID, D_K, D_POS, D_TAG
from utils.myutils import get_subfolder, load_pickle, store_pickle, update_dict


def query_net(net, states):
    """
    Call the net (brain) to determine the best suited action, given the stats (current environment observation)

    :param net: Network, created in one of the 'make_net' methods
    :param states: Current observations for each of the games
    """
    outputs = net.activate(states).numpy()
    return outputs


class CompleteExtinctionException(Exception):
    pass


class Population:
    """ Container for each of the agent's control mechanisms. """
    
    def __init__(self,
                 name: str = "",
                 folder_name: str = "NEAT",
                 version: int = 0,
                 config: NeatConfig = None,
                 make_net_method=make_net,
                 query_net_method=query_net,
                 ):
        """
        The population will be concerned about everything directly and solely related to the population. These are
        reporters, persisting methods, evolution and more.
        
        :param name: Name of population, used when specific population must be summon
        :param folder_name: Name of the folder to which the population belongs (NEAT, GRU-NEAT, ...)
        :param version: Version of the population, 0 if not versioned
        :param config: NeatConfig object
        :param make_net_method: Method used to create a network based on a given genome
        :param query_net_method: Method used to query the action on the network, given a current state
        """
        # Set formal properties of the population
        cfg = NeatConfig() if not config else config
        if name:
            self.name = name
        else:
            self.name = f"{cfg.fitness}{f'_{version}' if version else ''}"
        self.folder_name = folder_name
        
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
            self.create_population(cfg=cfg, make_net_method=make_net_method, query_net_method=query_net_method)
    
    def __str__(self):
        return self.name
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def create_population(self, cfg, make_net_method, query_net_method):
        """
        Create a new population based on the given config file.
        
        :param cfg: NeatConfig object
        :param make_net_method: Method used to create the genome-specific network
        :param query_net_method: Method used to query actions of the genome-specific network
        """
        # Init the population's configuration
        config = Config(
                genome_type=DefaultGenome,
                reproduction_type=DefaultReproduction,
                species_set_type=DefaultSpeciesSet,
                stagnation_type=DefaultStagnation,
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
        
        # Write population configuration to file
        with open(f'population/storage/{self.folder_name}/{self}/config.txt', 'w') as f:
            f.write(str(cfg))
            f.write("\n\n\n")  # 2 empty lines
            f.write(str(config))
    
    def evolve(self):
        """
        The evolution-process consists out of two phases:
          1) Reproduction
          2) Speciation
        
        This method manipulates the population itself, so nothing has to be returned
        """
        # Create the next generation from the current generation
        self.population = self.reproduction.reproduce(
                config=self.config,
                species=self.species,
                pop_size=self.config.pop_size,
                generation=self.generation,
        )
        
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
        save_path = get_subfolder(f'population/storage/{self.folder_name}/{self}/', 'images')
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
    
    def visualize_genome(self, debug=False, genome=None, name: str = '', show: bool = True):
        """
        Visualize the architecture of the given genome.
        
        :param debug: Add excessive genome-specific details in the plot
        :param genome: Genome that must be visualized, best genome is chosen if none
        :param name: Name of the image, excluding the population's generation (auto concatenated)
        :param show: Directly visualize the architecture
        """
        if not genome:
            genome = self.best_genome if self.best_genome else list(self.population.values())[0]
            if not name: name = 'best_genome_'
        name += 'gen_{gen:05d}'.format(gen=self.generation)
        get_subfolder(f'population/storage/{self.folder_name}/{self}/', 'images')
        sf = get_subfolder(f'population/storage/{self.folder_name}/{self}/images/', 'architectures')
        draw_net(self.config,
                 genome,
                 debug=debug,
                 filename=f'{sf}{name}',
                 view=show)
    
    # ---------------------------------------------> FUNCTIONAL METHODS <--------------------------------------------- #
    
    def add_evaluation_result(self, eval_result):
        """
        Append the result of the evaluation.
        
        :param eval_result: Dictionary
        """
        sf = get_subfolder(f'population/storage/{self.folder_name}/{self}/', 'evaluation')
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
        get_subfolder('population/storage/', f'{self.folder_name}')
        get_subfolder(f'population/storage/{self.folder_name}/', f'{self}')
        get_subfolder(f'population/storage/{self.folder_name}/{self}/', 'generations')
        
        # Save the population
        store_pickle(self, f'population/storage/{self.folder_name}/{self}/generations/gen_{self.generation:05d}')
        print(f"Population '{self}' saved! Current generation: {self.generation}")
    
    def load(self, gen=None):
        """
        Load in a game, specified by its current id.
        
        :return: True: game successfully loaded | False: otherwise
        """
        try:
            if gen is None:
                # Load in all previous populations
                populations = glob(f'population/storage/{self.folder_name}/{self}/generations/gen_*')
                if not populations: raise FileNotFoundError
                
                # Find newest population and save generation number under 'gen'
                populations = [p.replace('\\', '/') for p in populations]
                regex = r"(?<=" + \
                        re.escape(f'population/storage/{self.folder_name}/{self}/generations/gen_') + \
                        ")[0-9]*"
                gen = max([int(re.findall(regex, p)[0]) for p in populations])
            
            # Load in the population under the specified generation
            pop = load_pickle(f'population/storage/{self.folder_name}/{self}/generations/gen_{gen:05d}')
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
