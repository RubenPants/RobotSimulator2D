"""
evaluation_env.py

Evaluate a certain set of genomes on the evaluation mazes.
"""
import configparser
import multiprocessing as mp
import sys

from neat.six_util import iteritems

from utils.config import FPS
from utils.dictionary import D_DIST_TO_TARGET, D_DONE, D_STEPS

if sys.platform == 'linux':
    from environment.cy.multi_env_cy import MultiEnvironmentCy
else:
    from environment.multi_env import MultiEnvironment


class EvaluationEnv:
    """ This class is responsible evaluating the population across a set of games. """
    
    def __init__(self, rel_path=''):
        """
        The evaluator is given a set of genomes which it then evaluates using the MultiEnvironment.

        :param rel_path: Relative path pointing to the 'environment/' folder
        """
        # Set relative path
        self.rel_path = '{rp}{x}'.format(rp=rel_path, x='/' if (rel_path and rel_path[-1] not in ['/', '\\']) else '')
        
        # Load in current configuration
        self.config = configparser.ConfigParser()
        self.config.read('{}config.cfg'.format(self.rel_path))
        
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
            self.games = [i + 1 for i in range(int(self.config['GAME']['max_id']),
                                               int(self.config['GAME']['max_eval_id']))]
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
                    rel_path=self.rel_path,
                    max_duration=int(self.config['GAME']['duration'])
            )
        else:
            multi_env = MultiEnvironment(
                    make_net=pop.make_net,
                    query_net=pop.query_net,
                    rel_path=self.rel_path,
                    max_duration=int(self.config['GAME']['duration'])
            )
        
        # Evaluate on all the games
        multi_env.set_games(self.games)
        
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()
        
        def eval_genomes(genomes, config):
            for genome in genomes:
                processes.append(mp.Process(target=multi_env.eval_genome, args=(genome, config, return_dict)))
            
            for p in processes:
                p.start()
            
            for p in processes:
                p.join()
            
            # No need to validate the fitness in this scenario
            return return_dict
        
        # Evaluate the current population
        result_dict = eval_genomes(list(zip(range(len(genome_list)), genome_list)), pop.config)
        
        eval_result = dict()
        for k in result_dict.keys():
            eval_result[str(genome_list[k].key)] = create_answer(result_dict[k])
        pop.add_evaluation_result(eval_result)
    
    def evaluate_population(self, pop, game_ids=None):
        """
        Evaluate the population on a set of games and create blueprints of the final positions afterwards.

        :param pop: Population object
        :param game_ids: List of game-ids
        """
        if game_ids:
            self.set_games(game_ids)
        
        if len(self.games) > 20:
            raise Exception("It is not advised to evaluate a whole population on more than 20 games at once")
        
        if sys.platform == 'linux':
            multi_env = MultiEnvironmentCy(
                    make_net=pop.make_net,
                    query_net=pop.query_net,
                    rel_path=self.rel_path,
                    max_duration=int(self.config['GAME']['duration'])
            )
        else:
            multi_env = MultiEnvironment(
                    make_net=pop.make_net,
                    query_net=pop.query_net,
                    rel_path=self.rel_path,
                    max_duration=int(self.config['GAME']['duration'])
            )
        
        # Initialize the evaluation-pool
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # Evaluate the genomes
        for genome in list(iteritems(pop.population)):
            processes.append(mp.Process(target=multi_env.eval_genome, args=(genome, pop.config, return_dict)))
        
        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
        
        # Create blueprint of final result
        game_objects = [multi_env.create_game(g) for g in self.games]
        pop.create_blueprints(final_observations=return_dict, games=game_objects)


def create_answer(games):
    answer = ""
    answer += "Percentage finished: {p:.1f}".format(
            p=100 * len([g for g in games if g[D_DONE]]) / len(games))
    answer += " - Average distance to target {d:.1f}".format(
            d=sum([g[D_DIST_TO_TARGET] for g in games]) / len(games))
    answer += " - Max distance to target {d:.1f}".format(
            d=max([g[D_DIST_TO_TARGET] for g in games]))
    answer += " - Average time taken {t:.1f}".format(
            t=sum([g[D_STEPS] / FPS for g in games]) / len(games))
    answer += " - Max time taken {t:.1f}".format(
            t=max([g[D_STEPS] / FPS for g in games]))
    return answer
