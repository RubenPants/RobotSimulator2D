"""
env_visualizing.py

Visualize the performance of a population.
"""
import multiprocessing as mp
import sys

from neat.six_util import iteritems
from tqdm import tqdm

from config import GameConfig

if sys.platform == 'linux':
    from environment.cy.env_multi_cy import MultiEnvironmentCy
else:
    from environment.env_multi import MultiEnvironment


class VisualizingEnv:
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
        pool = mp.Pool(mp.cpu_count())
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # Fetch the dictionary of genomes
        genomes = list(iteritems(pop.population))
        
        # Progress bar during evaluation
        pbar = tqdm(total=len(genomes), desc="parallel training")
        
        def cb(*_):
            """Update progressbar after finishing a single genome's evaluation."""
            pbar.update()
        
        # Evaluate the genomes
        for genome in genomes:
            pool.apply_async(func=multi_env.eval_genome, args=(genome, pop.config, return_dict), callback=cb)
        pool.close()  # Close the pool
        pool.join()  # Postpone continuation until everything is finished
        
        # Create blueprint of final result
        game_objects = [multi_env.create_game(g) for g in self.games]
        pop.create_blueprints(final_observations=return_dict, games=game_objects)
