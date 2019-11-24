import configparser
import multiprocessing as mp
from random import sample

from neat.six_util import iteritems, itervalues

from environment.multi_env import MultiEnvironment


class Evaluator:
    """
    The evaluator is responsible evaluating the population across a set of games.
    """
    
    def __init__(self, rel_path=''):
        """
        TODO
        
        :param rel_path: Relative path pointing to the 'environment/' folder
        """
        # Set relative path
        self.rel_path = '{rp}{x}'.format(rp=rel_path, x='/' if (rel_path and rel_path[-1] not in ['/', '\\']) else '')
        
        # Load in current configuration
        self.config = configparser.ConfigParser()
        self.config.read('{}config.cfg'.format(self.rel_path))
        
        #  Create a list of all the possible games
        self.games = [i + 1 for i in range(int(self.config['GAME']['max_id']))]
        self.batch_size = min(len(self.games), int(self.config['GAME']['game_batch']))
    
    def single_evaluation(self, pop):
        """
        Evaluate the population for a single evaluation-process.
        """
        multi_env = MultiEnvironment(
                make_net=pop.make_net,
                query_net=pop.query_net,
                rel_path=self.rel_path,
                max_duration=int(self.config['GAME']['duration'])
        )
        
        # Set random set of games
        self.sample_games(multi_env)
        
        # Initialize the evaluation-pool
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
            
            # TODO: Check return_dict to obtain fitness scores
            for (_, genome) in genomes:
                genome.fitness = 0.5
        
        # Prepare the generation's reporters for the generation
        pop.reporters.start_generation(pop.generation)
        
        # Evaluate the current population
        eval_genomes(list(iteritems(pop.population)), pop.config)
        
        # Gather and report statistics
        best = None
        for g in itervalues(pop.population):
            if best is None or g.fitness > best.fitness:
                best = g
        pop.reporters.post_evaluate(pop.config, pop.population, pop.species, best)
        
        # Track best genome ever seen
        if pop.best_genome is None or best.fitness > pop.best_genome.fitness:
            pop.best_genome = best
        
        # Let population evolve
        pop.evolve()
        
        # End generation
        pop.reporters.end_generation(pop.config, pop.population, pop.species)
    
    def sample_games(self, multi_env):
        """
        Set the list on which the agents will be trained.
        
        :param multi_env: The environment on which the game-id list will be set
        """
        multi_env.set_games(sample(self.games, self.batch_size))
