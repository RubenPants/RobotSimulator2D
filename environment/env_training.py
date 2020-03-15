"""
env_training.py

Train and evaluate the population on a random batch of training games.
"""
import multiprocessing as mp
import sys
from random import sample

from neat.six_util import iteritems, itervalues
from tqdm import tqdm

from config import GameConfig
from population.population import Population
from population.utils.population_util.fitness_functions import calc_pop_fitness

if sys.platform == 'linux':
    from environment.cy.env_multi_cy import MultiEnvironmentCy
else:
    from environment.env_multi import MultiEnvironment


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
    
    def evaluate_and_evolve(self, pop: Population, n: int = 1, save_interval: int = 1, parallel=True):
        """
        Evaluate the population for a single evaluation-process.
        
        :param pop: Population object
        :param n: Number of generations
        :param save_interval: Indicates how often a population gets saved
        :param parallel: Parallel the code (disable parallelization for debugging purposes)
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
            self.sample_games(multi_env, pop.log)
            
            # Prepare the generation's reporters for the generation
            pop.reporters.start_generation(gen=pop.generation, logger=pop.log)
            
            # Initialize the evaluation-pool
            pool = mp.Pool(mp.cpu_count() - 2)  # TODO: Make sure laptop remains 'editable' during training
            manager = mp.Manager()
            return_dict = manager.dict()
            
            # Fetch the dictionary of genomes
            genomes = list(iteritems(pop.population))
            
            if parallel:
                pbar = tqdm(total=len(genomes), desc="parallel training")
                
                def cb(*_):
                    """Update progressbar after finishing a single genome's evaluation."""
                    pbar.update()
                
                for genome in genomes:
                    pool.apply_async(func=multi_env.eval_genome, args=(genome, pop.config, return_dict), callback=cb)
                pool.close()  # Close the pool
                pool.join()  # Postpone continuation until everything is finished
                pbar.close()  # Close the progressbar
                
                # Calculate the fitness from the given return_dict
                fitness = calc_pop_fitness(
                        fitness_config=pop.fitness_config,
                        game_observations=return_dict,
                        game_params=multi_env.get_game_params(),
                )
                for i, genome in genomes: genome.fitness = fitness[i]
            else:
                for genome in tqdm(genomes, desc="sequential training"):
                    multi_env.eval_genome(genome, pop.config, return_dict)
                fitness = calc_pop_fitness(
                        fitness_config=pop.fitness_config,
                        game_observations=return_dict,
                        game_params=multi_env.get_game_params(),
                )
                for i, genome in genomes: genome.fitness = fitness[i]
            
            # Gather and report statistics
            best = None
            for g in itervalues(pop.population):
                if best is None or g.fitness > best.fitness: best = g
            pop.reporters.post_evaluate(config=pop.config,
                                        population=pop.population,
                                        species=pop.species,
                                        best_genome=best,
                                        logger=pop.log)
            
            # Track best genome ever seen
            if pop.best_genome is None or best.fitness > pop.best_genome.fitness: pop.best_genome = best
            
            # Let population evolve
            pop.evolve()
            
            # End generation
            pop.reporters.end_generation(config=pop.config,
                                         population=pop.population,
                                         species_set=pop.species,
                                         logger=pop.log)
            
            # Save the population
            if iteration % save_interval == 0: pop.save()
    
    def sample_games(self, multi_env, logger=None):
        """
        Set the list on which the agents will be trained.
        
        :param multi_env: The environment on which the game-id list will be set
        :param logger: Log-reporter of population
        """
        s = sample(self.games, self.batch_size)
        msg = f"Sample chosen: {s}"
        logger(msg) if logger else print(msg)
        multi_env.set_games(s)
        return s
