import configparser
import multiprocessing as mp
from random import sample

from neat.six_util import iteritems, itervalues

from control.entities.fitness_functions import calc_pop_fitness
from environment.multi_env import MultiEnvironment


class Evaluator:
    """
    The evaluator is responsible evaluating the population across a set of games.
    """
    
    def __init__(self, blueprint_mazes: list = None, rel_path=''):
        """
        The evaluator is given a population which it then evaluates using the MultiEnvironment.
        
        :param blueprint_mazes: List of game-IDs for which a blueprint will be created after each generation
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
        
        # Persist the games for which blueprints must be created
        self.blueprint_mazes = blueprint_mazes
    
    def single_evaluation(self, pop):
        """
        Evaluate the population for a single evaluation-process.
        
        :param pop: Population object
        """
        # Create the environment which is responsible for evaluating the genomes
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
            
            # Calculate the fitness from the given return_dict
            fitness = calc_pop_fitness(fitness_config=pop.fitness_config, game_observations=return_dict)
            for i, genome in genomes:
                genome.fitness = fitness[i]
        
        # Prepare the generation's reporters for the generation
        pop.reporters.start_generation(pop.generation)
        
        # Evaluate the current population
        eval_genomes(list(iteritems(pop.population)), pop.config)
        
        # Create blueprint for selected games
        self.blueprint_genomes(population=pop)
        
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
        
        # Save the population
        pop.save()
    
    def sample_games(self, multi_env):
        """
        Set the list on which the agents will be trained.
        
        :param multi_env: The environment on which the game-id list will be set
        """
        s = sample(self.games, self.batch_size)
        print("Sample chose:", s)
        multi_env.set_games(s)
        return s
    
    def blueprint_genomes(self, population):
        """
        Create blueprints for all the requested mazes.
        
        :param population: Population that will be evaluated on the games
        :return:
        """
        multi_env = MultiEnvironment(
                make_net=population.make_net,
                query_net=population.query_net,
                rel_path=self.rel_path,
                max_duration=int(self.config['GAME']['duration'])
        )
        multi_env.set_games(self.blueprint_mazes)
        
        # Initialize the evaluation-pool
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # Evaluate the genomes
        for genome in list(iteritems(population.population)):
            processes.append(mp.Process(target=multi_env.eval_genome, args=(genome, population.config, return_dict)))
        
        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
        
        # Create blueprint of final result
        game_objects = [multi_env.create_game(g) for g in self.blueprint_mazes]
        population.create_blueprints(final_observations=return_dict, games=game_objects)
