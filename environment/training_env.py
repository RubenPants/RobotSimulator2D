"""
training_env.py

Train and evaluate the population on a random batch of training games.
"""
import multiprocessing as mp
import sys
from random import sample

from neat.six_util import iteritems, itervalues
from tqdm import tqdm

from configs.config import GameConfig
from population.fitness_functions import calc_pop_fitness

if sys.platform == 'linux':
    from environment.cy.multi_env_cy import MultiEnvironmentCy
else:
    from environment.multi_env import MultiEnvironment


class TrainingEnv:
    """ This class is responsible evaluating and evolving the population across a set of games. """
    
    __slots__ = (
        "cfg",
        "games", "batch_size",
    )
    
    def __init__(self):
        """ The evaluator is given a population which it then evaluates using the MultiEnvironment. """
        # Load in current configuration
        self.cfg: GameConfig = GameConfig()
        
        #  Create a list of all the possible games
        self.games = None
        self.batch_size = 0
        self.set_games()
    
    def set_games(self, games: list = None):
        """
        Set the game-IDs that will be used to evaluate the population. The full game-set as defined by the configuration
        file will be used if games=None.
        
        :param games: List of integers
        """
        if not games:
            self.games = [i + 1 for i in range(self.cfg.max_game_id)]
            self.batch_size = min(len(self.games), self.cfg.batch)
        else:
            self.games = games
            self.batch_size = len(games)
    
    def evaluate_and_evolve(self, pop, n: int = 1, save_interval: int = 1):
        """
        Evaluate the population for a single evaluation-process.
        
        :param pop: Population object
        :param n: Number of generations
        :param save_interval: Indicates how often a population gets saved
        """
        # Create the environment which is responsible for evaluating the genomes
        if sys.platform == 'linux':
            multi_env = MultiEnvironmentCy(
                    make_net=pop.make_net,
                    query_net=pop.query_net,
                    max_steps=self.cfg.duration * self.cfg.fps
            )
        else:
            multi_env = MultiEnvironment(
                    make_net=pop.make_net,
                    query_net=pop.query_net,
                    max_steps=self.cfg.duration * self.cfg.fps
            )
        
        for iteration in range(n):
            # Set random set of games
            self.sample_games(multi_env)
            
            # Initialize the evaluation-pool
            processes = []
            manager = mp.Manager()
            return_dict = manager.dict()
            
            def eval_genomes(genomes, config):
                for genome in genomes:
                    processes.append(mp.Process(target=multi_env.eval_genome, args=(genome, config, return_dict)))
                
                for p in tqdm(processes): p.start()
                for p in processes: p.join()
                
                # Calculate the fitness from the given return_dict
                fitness = calc_pop_fitness(fitness_config=pop.fitness_config, game_observations=return_dict)
                for i, genome in genomes:
                    genome.fitness = fitness[i]
            
            # Prepare the generation's reporters for the generation
            pop.reporters.start_generation(pop.generation)
            
            # Evaluate the current population
            eval_genomes(list(iteritems(pop.population)), pop.config)
            
            # Gather and report statistics
            best = None
            for g in itervalues(pop.population):
                if best is None or g.fitness > best.fitness: best = g
            pop.reporters.post_evaluate(pop.config, pop.population, pop.species, best)
            
            # Track best genome ever seen
            if pop.best_genome is None or best.fitness > pop.best_genome.fitness: pop.best_genome = best
            
            # Let population evolve
            pop.evolve()
            
            # End generation
            pop.reporters.end_generation(pop.config, pop.population, pop.species)
            
            # Save the population
            if iteration % save_interval == 0: pop.save()
    
    def sample_games(self, multi_env):
        """
        Set the list on which the agents will be trained.
        
        :param multi_env: The environment on which the game-id list will be set
        """
        s = sample(self.games, self.batch_size)
        print("Sample chosen:", s)
        multi_env.set_games(s)
        return s
    
    def blueprint_genomes(self, pop):
        """
        Create blueprints for all the requested mazes.
        
        :param pop: Population object
        """
        if sys.platform == 'linux':
            multi_env = MultiEnvironmentCy(
                    make_net=pop.make_net,
                    query_net=pop.query_net,
                    max_steps=self.cfg.duration * self.cfg.fps
            )
        else:
            multi_env = MultiEnvironment(
                    make_net=pop.make_net,
                    query_net=pop.query_net,
                    max_steps=self.cfg.duration * self.cfg.fps
            )
        
        if len(self.games) > 20:
            raise Exception("It is not advised to evaluate on more than 20 at once")
        
        multi_env.set_games(self.games)
        
        # Initialize the evaluation-pool
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # Evaluate the genomes
        for genome in list(iteritems(pop.population)):
            processes.append(mp.Process(target=multi_env.eval_genome, args=(genome, pop.config, return_dict)))
        
        for p in tqdm(processes): p.start()
        for p in processes: p.join()
        
        # Create blueprint of final result
        game_objects = [multi_env.create_game(g) for g in self.games]
        pop.create_blueprints(final_observations=return_dict, games=game_objects)
