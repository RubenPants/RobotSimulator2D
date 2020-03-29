"""
env_training.py

Train and evaluate the population on a random batch of training games.
"""
import multiprocessing as mp
import sys
import traceback
import warnings
from random import sample

from neat.six_util import iteritems, itervalues
from tqdm import tqdm

from config import Config
from environment.env_multi import get_multi_env
from population.population import Population
from population.utils.population_util.fitness_functions import calc_pop_fitness

if sys.platform == 'linux':
    pass
else:
    pass


class TrainingEnv:
    """ This class is responsible evaluating and evolving the population across a set of games. """
    
    __slots__ = (
        "game_config", "unused_cpu",
        "games", "batch_size",
    )
    
    def __init__(self, game_config: Config, unused_cpu: int = 0):
        """ The evaluator is given a population which it then evaluates using the MultiEnvironment. """
        # Load in current configuration
        self.game_config = game_config
        self.unused_cpu = unused_cpu
        
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
            self.games = [i + 1 for i in range(self.game_config.game.max_game_id)]
            self.batch_size = min(len(self.games), self.game_config.game.batch)
        else:
            self.games = games
            self.batch_size = len(games)
    
    def evaluate_and_evolve(self,
                            pop: Population,
                            n: int = 1,
                            parallel=True,
                            save_interval: int = 1,
                            ):
        """
        Evaluate the population for a single evaluation-process.
        
        :param pop: Population object
        :param n: Number of generations
        :param parallel: Parallel the code (disable parallelization for debugging purposes)
        :param save_interval: Indicates how often a population gets saved
        """
        multi_env = get_multi_env(pop=pop, game_config=self.game_config)
        saved = True
        for iteration in range(n):
            # Set random set of games
            self.sample_games(multi_env, pop.log)
            
            # Evaluate the population on the newly sampled games
            single_evaluation(multi_env=multi_env,
                              parallel=parallel,
                              pop=pop,
                              unused_cpu=self.unused_cpu,
                              )
            
            # Save the population
            if (iteration + 1) % save_interval == 0:
                pop.save()
                saved = True
            else:
                saved = False
        
        # Make sure that last iterations saves
        if not saved: pop.save()
    
    def evaluate_same_games_and_evolve(self,
                                       games: list,
                                       pop: Population,
                                       n: int = 1,
                                       parallel=True,
                                       save_interval: int = 1,
                                       ):
        """
        Evaluate the population on the same games.
        
        :param games: List of games used for training
        :param pop: Population object
        :param n: Number of generations
        :param parallel: Parallel the code (disable parallelization for debugging purposes)
        :param save_interval: Indicates how often a population gets saved
        """
        multi_env = get_multi_env(pop=pop, game_config=self.game_config)
        msg = f"Repetitive evaluating games: {games}"
        pop.log(msg, print_result=False)
        multi_env.set_games(games)
        
        # Iterate and evaluate over the games
        saved = True
        for iteration in range(n):
            single_evaluation(multi_env=multi_env,
                              parallel=parallel,
                              pop=pop,
                              unused_cpu=self.unused_cpu,
                              )
            
            # Save the population
            if (iteration + 1) % save_interval == 0:
                pop.save()
                saved = True
            else:
                saved = False
        
        # Make sure that last iterations saves
        if not saved: pop.save()
    
    def sample_games(self, multi_env, logger=None):
        """
        Set the list on which the agents will be trained.
        
        :param multi_env: The environment on which the game-id list will be set
        :param logger: Log-reporter of population
        """
        s = sample(self.games, self.batch_size)
        msg = f"Sample chosen: {s}"
        logger(msg, print_result=False) if logger else print(msg)
        multi_env.set_games(s)
        return s


def single_evaluation(multi_env, parallel: bool, pop: Population, unused_cpu: int):
    """
    Perform a single evaluation-iteration.
    
    :param multi_env: Environment used to execute the game-simulation in
    :param parallel: Boolean indicating if training happens in parallel or not
    :param pop: Population used to evaluate on
    :param unused_cpu: Number of CPU-cores not used during evaluation
    """
    # Prepare the generation's reporters for the generation
    pop.reporters.start_generation(gen=pop.generation, logger=pop.log)
    
    # Fetch the dictionary of genomes
    genomes = list(iteritems(pop.population))
    
    if parallel:
        pbar = tqdm(total=len(genomes), desc="parallel training")
        
        # Initialize the evaluation-pool
        pool = mp.Pool(mp.cpu_count() - unused_cpu)
        manager = mp.Manager()
        return_dict = manager.dict()
        
        def cb(*_):
            """Update progressbar after finishing a single genome's evaluation."""
            pbar.update()
        
        for genome in genomes:
            pool.apply_async(func=multi_env.eval_genome, args=(genome, return_dict), callback=cb)
        pool.close()  # Close the pool
        pool.join()  # Postpone continuation until everything is finished
        pbar.close()  # Close the progressbar
    else:
        return_dict = dict()
        for genome in tqdm(genomes, desc="sequential training"):
            multi_env.eval_genome(genome, return_dict)
    
    # Calculate the fitness from the given return_dict
    try:
        fitness = calc_pop_fitness(
                fitness_config=pop.config.evaluation,
                game_observations=return_dict,
                game_params=multi_env.get_game_params(),
                generation=pop.generation,
        )
        for i, genome in genomes: genome.fitness = fitness[i]
    except Exception:  # TODO: Fix! Sometimes KeyError in fitness (path problem)
        pop.log(f"Exception at fitness calculation: \n{traceback.format_exc()}", print_result=False)
        warnings.warn(f"Exception at fitness calculation: \n{traceback.format_exc()}")
        # Set fitness to zero for genomes that have no fitness set yet
        for i, genome in genomes:
            if not genome.fitness: genome.fitness = 0.0
    
    # Gather and report statistics
    best = None
    for g in itervalues(pop.population):
        if best is None or g.fitness > best.fitness: best = g
    pop.reporters.post_evaluate(population=pop.population,
                                species=pop.species,
                                best_genome=best,
                                logger=pop.log)
    
    # Update the population's best_genome
    genomes = sorted(pop.population.items(), key=lambda x: x[1].fitness, reverse=True)
    pop.best_genome_hist[pop.generation] = genomes[:pop.config.population.genome_elitism]
    if pop.best_genome is None or best.fitness > pop.best_genome.fitness: pop.best_genome = best
    
    # Let population evolve
    pop.evolve()
    
    # End generation
    pop.reporters.end_generation(population=pop.population,
                                 name=str(pop),
                                 species_set=pop.species,
                                 logger=pop.log)
