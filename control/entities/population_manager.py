"""
population_manager.py

A population is used as a container for all the current and past genomes of its candidates.
This implementation will assign a separate folder for each population, which contains the whole history of this
population (saved as populations themselves).
"""
import pickle
import re
import shutil
from configparser import ConfigParser
from glob import glob

import neat

from control.entities.population import Population
from environment.environment import Environment
from pytorch_neat.neat_reporter import LogReporter
from utils.config import *
from utils.dictionary import *
from utils.myutils import drop, get_subfolder, prep


class PopulationManager:
    def __init__(self,
                 generation: int = None,
                 make_net_method=None,
                 name: str = None,
                 query_net_method=None,
                 rel_path: str = '',
                 silent: bool = False):
        """
        Container for the population.
        
        :param generation: Load in a specific generation if not None
        :param make_net_method: Method used to create a network based on a given genome
        :param name: Name of the population
        :param query_net_method: Method used to query the action on the network, given a current state
        :param rel_path: Relative path to folder in which population is invoked (e.g. control/NEAT/)
        :param silent: Do not print information
        """
        # Functional parameters (population independent)
        self.rel_path = '{rp}{x}'.format(rp=rel_path, x='/' if (rel_path and rel_path[-1] not in ['/', '\\']) else '')
        self.silent = silent
        
        # Population specific
        self.config = None
        self.name = name if name else "population_{nr:05d}".format(
                nr=len(glob(get_subfolder(self.rel_path, 'populations') + '*')))
        self.population = None
        self.logger = None
        
        # Create subfolder (contains all generations of this population) if not exists
        get_subfolder('{}populations/'.format(self.rel_path), str(self))
        
        # Environment specific
        self.environment: Environment = None
        
        # Network methods
        self.make_net_method = make_net_method
        self.query_net_method = query_net_method
        
        # Load in the game-IDs
        self.empty_maze = [0]
        self.t_maze = [99999]
        self.random_mazes = [i + 1 for i in range(GAMES_AMOUNT)]
        
        # Load in the
        if not self.load(gen=generation):
            self.create_init_population()
    
    def __str__(self):
        """
        Name of the population.
        """
        return self.name
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def create_init_population(self):
        """
        Create the initial population based on the provided config-file.
        """
        # Check first if all necessary attributes are provided
        s = []
        if not self.make_net_method: s.append('make_net_method')
        if not self.query_net_method: s.append('query_net_method')
        if s: raise Exception("No population was created, the following argument{s} must be given: {args}"
                              .format(s='s' if len(s) > 1 else '', args=', '.join(s)))
        
        # Setup the folder
        get_subfolder(get_subfolder(self.rel_path, 'populations'), str(self))
        
        # Copy the config-file to the folder
        shutil.copy('{rp}config.cfg'.format(rp=self.rel_path),
                    '{rp}populations/{p}/config.cfg'.format(rp=self.rel_path, p=self))
        
        # Setup the config file
        self.config = self.get_config()
        
        # The environments used
        cfg = ConfigParser()
        cfg.read('{rp}populations/{p}/config.cfg'.format(rp=self.rel_path, p=self))
        fitness_config = {
            D_FIT_COMB: cfg['EVALUATION']['fitness_comb'],
            D_K:        int(cfg['EVALUATION']['nn_k']),
            D_TAG:      cfg['EVALUATION']['fitness'],
        }
        self.environment = Environment(
                fitness_config=fitness_config,
                make_net=self.make_net,
                max_duration=int(cfg['EVALUATION']['game_duration']),
                max_batch_size=int(cfg['EVALUATION']['batch_size']),
                query_net=self.query_net,
                rel_path='../../environment/'
        )
        
        # The population represents the core of evolution algorithm:
        #  1. Evaluate the fitness of all the genomes
        #  2. Check to see if termination criterion is satisfied, exit if so
        #  3. Generate the next generation from the current population
        #  4. Partition the new generation into species based on genetic similarity
        #  5. Go back to 1.
        self.population = Population(config=self.config,
                                     game_path='{rp}../../environment/'.format(rp=self.rel_path),
                                     save_blueprint_path=get_subfolder(
                                             '{rp}populations/{s}/'.format(rp=self.rel_path, s=self),
                                             'blueprints'))
        
        # The StatisticsReporter gathers and provides the most-fit genomes and information on genome and species fitness
        # and species sizes
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        
        # Use 'print' to output information about the run
        reporter = neat.StdOutReporter(True)
        self.population.add_reporter(reporter)
        
        # Write each generation-overview to a log-file
        self.logger = None
        self.create_log_reporter()
        
        if not self.silent:
            print("Population '{}' successful created!".format(self))
        
        # Save the current (initial) population
        self.save()
    
    def evaluate(self, n_generations: int = None):
        """
        Evaluate and evolve the population over n generations.

        :param n_generations: Number of generations over which the population has evolved, 100 000 if None
        """
        self.population.run(calculate_fitness=self.environment.calculate_fitness,
                            eval_generation=self.environment.eval_generation,
                            n_gen=n_generations,
                            save_f=self.save)
    
    def visualize(self, debug: bool = False, genome_key: int = None, game_id: int = 0, speedup: int = 1):
        """
        Evaluate a genome, specified by its ID, of the current population.
        
        :param debug: Print out actions of agent
        :param game_id: Identification number of the game that will be used for evaluation
        :param genome_key: Int: Key of the requested genome | None: current best genome
        :param speedup: Relative speed of the game to the real world
        :return: Performance Dictionary
        """
        # Set the game
        self.environment.game_id = game_id
        self.environment.play_speedup = speedup
        
        # Load in the genome
        if genome_key and genome_key not in self.population.population.keys():
            raise KeyError("Invalid genome key, choose from {}".format(self.population.population.keys()))
        genome = self.population.population[genome_key] if genome_key else self.population.best_genome
        
        # Run the game
        raise NotImplemented  # TODO: Create visualization!
        self.environment.visualize_genome(genome, self.config, debug=debug)
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def create_log_reporter(self):
        """
        Create the log-reporter which will write logs in the rel_path root folder during training.
        """
        self.logger = LogReporter("{rp}populations/{s}/log_file.log".format(rp=self.rel_path, s=self),
                                  self.environment.eval_genome)
        self.population.add_reporter(self.logger)
    
    def delete_log_reporter(self):
        """
        Remove the log-reporter from the population since this is not serializable.
        """
        self.population.remove_reporter(self.logger)
        self.logger = None
    
    def evaluate_chosen_maze(self, n_generations, maze_id: int):
        """
        Evaluate and evolve the population over n generations on the empty maze.
        
        :param maze_id: ID of the maze chosen to evaluate on
        :param n_generations: Number of generations over which the population has evolved
        """
        self.environment.set_game_list([maze_id])
        self.evaluate(n_generations)
    
    def evaluate_empty_maze(self, n_generations):
        """
        Evaluate and evolve the population over n generations on the empty maze.
        
        :param n_generations: Number of generations over which the population has evolved
        """
        self.environment.set_game_list(self.empty_maze)
        self.evaluate(n_generations)
    
    def evaluate_random_mazes(self, n_generations: int = None):
        """
        Evaluate and evolve the population over n generations on the randomly created mazes.
        
        :param n_generations: Number of generations over which the population has evolved
        """
        self.environment.set_game_list(self.random_mazes)
        self.evaluate(n_generations)
    
    def evaluate_t_maze(self, n_generations):
        """
        Evaluate and evolve the population over n generations on the T-maze.
        
        :param n_generations: Number of generations over which the population has evolved
        """
        self.environment.set_game_list(self.t_maze)
        self.evaluate(n_generations)
    
    def get_config(self):
        """
        Get the config-file stored in the local (population) directory.
        """
        return neat.Config(
                genome_type=neat.DefaultGenome,
                reproduction_type=neat.DefaultReproduction,
                species_set_type=neat.DefaultSpeciesSet,
                stagnation_type=neat.DefaultStagnation,
                filename="{rp}populations/{s}/config.cfg".format(rp=self.rel_path, s=self),
        )
    
    def make_net(self, genome, config, bs):
        """
        Create the "brains" of the candidate, based on its genetic wiring.
    
        :param genome: Genome specifies the brains internals
        :param config: Configuration file from this folder: config.cfg
        :param bs: Batch size, which represents amount of games trained in parallel
        """
        return self.make_net_method(genome, config, bs)
    
    def query_net(self, net, state):
        """
        Call the net (brain) to determine the best suited action, given the stats (current environment observation)
    
        :param net: RecurrentNet, created in 'make_net' (previous method)
        :param state: Current observations
        """
        return self.query_net_method(net, state)
    
    # ---------------------------------------------> FUNCTIONAL METHODS <--------------------------------------------- #
    
    def save(self):
        """
        Save the population as the current generation.
        """
        if TIME_ALL:
            prep(key='load_save', silent=True)
        try:
            # Create needed subfolder if not yet exist
            get_subfolder(self.rel_path, 'populations')
            get_subfolder('{}populations/'.format(self.rel_path), '{}'.format(self))
            get_subfolder('{rp}populations/{pop}/'.format(rp=self.rel_path, pop=self), 'generations')
            
            # Remove log-reporter since not serializable
            self.delete_log_reporter()
            
            # Save the population
            pickle.dump(self, open('{rp}populations/{pop}/generations/gen_{gen:05d}'.format(
                    rp=self.rel_path,
                    pop=self,
                    gen=self.population.generation),
                    'wb'))
            # Restore removed log-reporter
            self.create_log_reporter()
        finally:
            if not self.silent:
                print("Population '{}' saved!".format(self))
            if TIME_ALL:
                drop(key='load_save', silent=True)
    
    def load(self, gen=None):
        """
        Load in a game, specified by its current id.
        
        :return: True: game successfully loaded | False: otherwise
        """
        if TIME_ALL:
            prep(key='load_save', silent=True)
        try:
            if not gen:
                # Load in all previous populations
                populations = glob('{rp}populations/{pop}/generations/gen_*'.format(rp=self.rel_path, pop=self))
                if not populations: raise FileNotFoundError
                
                # Find newest population and save generation number under 'gen'
                populations = [p.replace('\\', '/') for p in populations]
                regex = r"(?<=" + \
                        re.escape('{rp}populations/{pop}/generations/gen_'.format(rp=self.rel_path, pop=self)) + \
                        ")[0-9]*"
                gen = max([int(re.findall(regex, p)[0]) for p in populations])
            
            # Load in the population under the specified generation
            pop = pickle.load(open('{rp}populations/{pop}/generations/gen_{gen:05d}'.format(rp=self.rel_path,
                                                                                            pop=self,
                                                                                            gen=gen), 'rb'))
            self.config = self.get_config()
            self.environment = pop.environment
            self.make_net_method = pop.make_net
            self.population = pop.population
            self.population.set_config(self.config)
            self.query_net_method = pop.query_net
            self.create_log_reporter()  # add log-reporter
            if not self.silent:
                print("Population {} loaded successfully!".format(self))
            return True
        except FileNotFoundError:
            return False
        finally:
            if TIME_ALL:
                drop(key='load_save', silent=True)
