"""
evaluation_env.py

Evaluate a certain set of genomes on the evaluation mazes.
"""
import multiprocessing as mp
import sys

from neat.six_util import iteritems
from tqdm import tqdm

from config import GameConfig
from utils.dictionary import D_DIST_TO_TARGET, D_DONE, D_STEPS

if sys.platform == 'linux':
    from environment.cy.multi_env_cy import MultiEnvironmentCy
else:
    from environment.multi_env import MultiEnvironment


class EvaluationEnv:
    """ This class is responsible evaluating the population across a set of games. """
    
    __slots__ = (
        "cfg",
        "games", "batch_size",
    )
    
    def __init__(self):
        """ The evaluator is given a set of genomes which it then evaluates using the MultiEnvironment. """
        # Load in current configuration
        self.cfg: GameConfig = GameConfig()
        
        #  Create a list of all the possible games
        self.games = None
        self.batch_size = 0
    
    def set_games(self, games: list = None):
        """
        Set the game-IDs that will be used to evaluate the population. The full game-set as defined by the configuration
        file will be used if games=None.
        
        :param games: List of integers
        """
        if not games:
            self.games = [i + 1 for i in range(self.cfg.max_game_id, self.cfg.max_eval_game_id)]
            self.batch_size = len(self.games)
        else:
            self.games = games
            self.batch_size = len(games)
    
    def evaluate_genome_list(self, genome_list, pop):
        """
        Evaluate the population for a single evaluation-process.

        :param genome_list: List of genomes
        :param pop: The population to which the genomes belong (used to setup the network and query the config)
        """
        # Evaluate on all the evaluation games
        self.set_games()
        
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
        
        # Evaluate on all the games
        multi_env.set_games(self.games)
        
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
        
        for genome in genomes:
            pool.apply_async(func=multi_env.eval_genome, args=(genome, pop.config, return_dict), callback=cb)
        pool.close()  # Close the pool
        pool.join()  # Postpone continuation until everything is finished
        pbar.close()  # Close the progressbar
        
        eval_result = dict()
        for k in return_dict.keys(): eval_result[str(genome_list[k].key)] = create_answer(return_dict[k])
        pop.add_evaluation_result(eval_result)
    
    def evaluate_population(self, pop, game_ids=None):
        """
        Evaluate the population on a set of games and create blueprints of the final positions afterwards.

        :param pop: Population object
        :param game_ids: List of game-ids
        """
        if game_ids: self.set_games(game_ids)
        
        if len(self.games) > 20:
            raise Exception("It is not advised to evaluate a whole population on more than 20 games at once")
        
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
        pbar.close()  # Close the progressbar
        
        # Create blueprint of final result
        game_objects = [multi_env.create_game(g) for g in self.games]
        pop.create_blueprints(final_observations=return_dict, games=game_objects)


def create_answer(games: list):
    cfg = GameConfig()
    answer = ""
    answer += f"Percentage finished: {100 * len([g for g in games if g[D_DONE]]) / len(games):.1f}"
    answer += f" - Average distance to target {sum([g[D_DIST_TO_TARGET] for g in games]) / len(games):.1f}"
    answer += f" - Max distance to target {max([g[D_DIST_TO_TARGET] for g in games]):.1f}"
    answer += f" - Average time taken {sum([g[D_STEPS] / cfg.fps for g in games]) / len(games):.1f}"
    answer += f" - Max time taken {max([g[D_STEPS] / cfg.fps for g in games]):.1f}"
    return answer
