"""
environment.py

Handle the communication between the population and the game. The environment is used to evaluate multiple games in
parallel for each of the agents.
"""
from random import sample

from tqdm import tqdm

from control.entities.fitness_functions import fitness
from environment.entities.game import Game
from utils.config import FPS
from utils.dictionary import D_SENSOR_LIST


class EnvironmentEvaluator:
    """
    Environment which supports the evaluation of multiple games in parallel. The batch_size defines the number of
    environments on which one single candidate is trained.
    """
    
    def __init__(self,
                 fitness_config: dict,
                 make_net,
                 max_duration: int,
                 query_net,
                 game_list: list = None,
                 max_batch_size: int = 16,
                 rel_path: str = ''):
        """
        Initialize the environment used for training. This environment will be purely from CLI, so no visual feedback
        except printouts are given.

        :param fitness_config: Dictionary containing all the configuration needed to setup the fitness-function
        :param make_net: Method to create a network based on the given genome
        :param max_duration: Maximum number of seconds a candidate drives around in a single environment
        :param query_net: Method to evaluate the network given the current state
        :param game_list: List of integers of game-IDs which are used to to train the population on
        :param max_batch_size: Maximum number of mazes evaluated on in parallel
        :param rel_path: Path pointing to this directory ('environment/')
        """
        # General environment parameters
        self.make_net = make_net
        self.max_steps = max_duration * FPS
        self.query_net = query_net
        self.rel_path = rel_path
        
        # Environment specific parameters
        self.batch_size = None  # Number of games considered during one evaluation session
        self.game_id = None  # Used for game-visualization (only one game possible)
        self.game_id_list = None  # Used for sequential game-evaluation
        self.fitness_cfg = fitness_config  # Configuration file for the fitness-function
        self.max_batch_size = max_batch_size  # Maximum possible batch_size
        self.play_speedup = 1  # Relative speedup for the game visualization
        self.set_game_list(game_list)  # Init the game_list
        
        # Un-initialized parameters
        self.eval_game_ids = None  # Container for all the games used to evaluate the candidates on
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def calculate_fitness(self, genomes, observations: dict):
        """
        Calculate the fitness for each of the genomes given the game.close() observation list.
        
        :param genomes: Collection of genomes for each of the candidates in the population
        :param observations: Dictionary with { genome_key : [game.close()] }
        """
        # Fetch and evaluate requested fitness
        total_fitness = fitness(fitness_config=self.fitness_cfg,
                                game_observations=observations)
        
        # Apply fitness to each of the genomes
        for i, genome in genomes:
            genome.fitness = total_fitness[i]
    
    def eval_generation(self, genomes, config):
        """
        Test a complete generation on a random subset of game.
        
        :param genomes: Collection of genomes for each of the candidates in the population
        :param config: Configuration file specifying the candidates
        :return: Dictionary with { genome_key : [game.close()] } and list of game-IDs used for evaluation
        """
        # Determine random game-set
        self.eval_game_ids = sample(self.game_id_list, self.batch_size)
        
        # Evaluate the generation
        observations = dict()
        for i, genome in tqdm(genomes, desc="Evaluating genomes"):
            observations[i] = self.eval_genome(genome, config)
        
        return observations, self.eval_game_ids
    
    def eval_genome(self, gen, cfg, debug=False):
        """
        Evaluate a single genome on a set of games.
    
        :param gen: The genome of the candidate
        :param cfg: The main configuration of the population
        :return: Final game statistics
        """
        # Setup game
        games = [self.create_game(i) for i in self.eval_game_ids]
        
        # Setup the candidate
        net = self.make_net(gen, cfg, self.batch_size)
        
        # Placeholders
        states = [g.reset()[D_SENSOR_LIST] for g in games]
        finished = [False] * self.batch_size
        
        # Iterate in the environment until time has passed
        step_num = 0
        while True:
            step_num += 1
            
            # Check if maximum of iterations reached
            if self.max_steps is not None and step_num == self.max_steps:
                break
            
            # Determine the actions via the provided function
            if debug:
                actions = self.query_net(net, states, debug=True, step_num=step_num)
            else:
                actions = self.query_net(net, states)
            
            assert len(actions) == len(games)
            
            # Iterate over all of the environments to progress one step
            for i, (game, action, done) in enumerate(zip(games, actions, finished)):
                # Ignore if environment is put on 'done'
                if not done:
                    # Progress the environment by one step
                    observation, finished[i] = game.step(l=action[0], r=action[1])
                    
                    # Update the candidate's current state
                    states[i] = observation[D_SENSOR_LIST]
            
            # If all the environments are finished, break out of the "while True" loop
            if all(finished):
                break
        
        # Return the final observations
        return [g.close() for g in games]
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def create_game(self, i):
        """
        :param i: Game-ID
        :return: Game object
        """
        return Game(game_id=i,
                    rel_path=self.rel_path,
                    silent=True)
    
    def set_game_list(self, game_list: list = None):
        """
        
        :param game_list:
        :return:
        """
        self.batch_size = min(self.max_batch_size, len(game_list)) if game_list else 1
        self.game_id_list = game_list if game_list else [0]
