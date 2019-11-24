import multiprocessing as mp

from neat.six_util import iteritems, itervalues

from environment.multi_env import MultiEnvironment


class Evaluator:
    """
    The evaluator is responsible evaluating the population across a set of games.
    """
    
    def __init__(self,
                 config=None,
                 rel_path=''):
        """
        TODO
        
        :param config:
        :param rel_path:
        """
        self.rel_path = rel_path
        self.config = config  # TODO: Needed? For what? Separate game-config perhaps?
        self.games = [1, 2, 3, 4]  # TODO: init list of all possible game-ids
    
    def single_evaluation(self, pop):
        """
        Evaluate the population for a single evaluation-process.
        """
        multi_env = MultiEnvironment(
                make_net=pop.make_net,
                query_net=pop.query_net,  # TODO: Parse duration from config!
                rel_path=self.rel_path,
        )
        
        # Set random set of games
        self.sample_games(multi_env)
        
        # Initialize the evaluation-pool
        # pool = mp.Pool(mp.cpu_count())
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
        TODO: Sample random subset of game-ids and create game-objects
        :param multi_env:
        :return:
        """
        # TODO: Sample first on self.games!
        multi_env.set_games(self.games)  # TODO
