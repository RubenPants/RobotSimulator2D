"""
evaluation_env.py

Evaluate a certain set of genomes on the evaluation mazes.
"""
import multiprocessing as mp

from neat.six_util import iteritems
from tqdm import tqdm

from config import Config
from environment.entities.game import get_game
from environment.env_training import get_multi_env
from population.utils.visualizing.population_visualizer import create_blueprints
from utils.dictionary import D_DIST_TO_TARGET, D_DONE, D_TIME_TAKEN
from utils.myutils import get_subfolder


class EvaluationEnv:
    """ This class is responsible evaluating the population across a set of games. """
    
    __slots__ = (
        "game_config",
        "games", "batch_size",
    )
    
    def __init__(self, game_config: Config):
        """ The evaluator is given a set of genomes which it then evaluates using the MultiEnvironment. """
        # Load in current configuration
        self.game_config = game_config
        
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
            self.games = [i + 1 for i in range(self.game_config.game.max_game_id,
                                               self.game_config.game.max_eval_game_id)]
            self.batch_size = len(self.games)
        else:
            self.games = games
            self.batch_size = len(games)
    
    def evaluate_genome_list(self, genome_list, pop):
        """
        Evaluate the population for a single evaluation-process.

        :param genome_list: List of genomes that will be evaluated
        :param pop: The population to which the genomes belong (used to setup the network and query the config)
        """
        # Evaluate on all the evaluation games
        self.set_games()
        
        # Create the environment which is responsible for evaluating the genomes
        multi_env = get_multi_env(pop=pop, game_config=self.game_config)
        
        # Evaluate on all the games
        multi_env.set_games(self.games)
        
        pool = mp.Pool(mp.cpu_count())
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # Fetch requested genomes
        genomes = [(g.key, g) for g in genome_list]
        
        # Progress bar during evaluation
        pbar = tqdm(total=len(list(genomes)), desc="parallel evaluating")
        
        def cb(*_):
            """Update progressbar after finishing a single genome's evaluation."""
            pbar.update()
        
        for genome in genomes:
            pool.apply_async(func=multi_env.eval_genome, args=(genome, return_dict), callback=cb)
        pool.close()  # Close the pool
        pool.join()  # Postpone continuation until everything is finished
        pbar.close()  # Close the progressbar
        
        eval_result = dict()
        for k in return_dict.keys(): eval_result[str(k)] = create_answer(return_dict[k])
        pop.add_evaluation_result(eval_result)
    
    def evaluate_population(self, pop, game_ids=None):  # TODO: Not used, remove?
        """
        Evaluate the population on a set of games and create blueprints of the final positions afterwards.

        :param pop: Population object
        :param game_ids: List of game-ids
        """
        if game_ids: self.set_games(game_ids)
        
        if len(self.games) > 20:
            raise Exception("It is not advised to evaluate a whole population on more than 20 games at once")
        
        # Create the environment which is responsible for evaluating the genomes
        multi_env = get_multi_env(pop=pop, game_config=self.game_config)
        
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
            pool.apply_async(func=multi_env.eval_genome, args=(genome, return_dict), callback=cb)
        pool.close()  # Close the pool
        pool.join()  # Postpone continuation until everything is finished
        pbar.close()  # Close the progressbar
        
        # Create blueprint of final result
        game_objects = [get_game(g, cfg=self.game_config) for g in self.games]
        create_blueprints(final_observations=return_dict,
                          games=game_objects,
                          gen=pop.generation,
                          save_path=get_subfolder(f'population/storage/{pop.folder_name}/{pop}/', 'images'))


def create_answer(games: list):
    cfg = Config()
    answer = dict()
    answer['Percentage finished'] = round(100 * len([g for g in games if g[D_DONE]]) / len(games), 2)
    
    answer['Min distance to target'] = round(min([g[D_DIST_TO_TARGET] for g in games]), 2)
    answer['Average distance to target'] = round(sum([g[D_DIST_TO_TARGET] for g in games]) / len(games), 2)
    answer['Max distance to target'] = round(max([g[D_DIST_TO_TARGET] for g in games]), 2)
    
    answer['Min time taken'] = round(min([g[D_TIME_TAKEN] for g in games]), 2)
    answer['Average time taken'] = round(sum([g[D_TIME_TAKEN] for g in games]) / len(games), 2)
    answer['Max time taken'] = round(max([g[D_TIME_TAKEN] for g in games]), 2)
    return answer
