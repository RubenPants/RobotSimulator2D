"""
population.py

Copy and extension of neat.Population, which implements the core evolution algorithm.
"""
from __future__ import print_function

import matplotlib.pyplot as plt
from neat.math_util import mean
from neat.reporting import ReporterSet
from neat.six_util import iteritems, itervalues

from environment.entities.game import Game
from utils.config import *
from utils.dictionary import *


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """
    
    def __init__(self,
                 config,
                 game_path: str = '',
                 initial_state=None,
                 save_blueprint_path: str = ''):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config,
                                            self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        
        # Fitness evaluation
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                    "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))
        
        if initial_state is None:
            # Create a population from scratch, then partition into species
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config,
                                                   self.reporters)
            self.generation = 0
            self.species.speciate(config,
                                  self.population,
                                  self.generation)
        else:
            self.population, self.species, self.generation = initial_state
        
        # Placeholder for the most fit genome
        self.best_genome = None
        
        # Other parameters
        self.save_blueprint_path = save_blueprint_path
        self.rel_game_path = game_path
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def run(self, calculate_fitness, eval_generation, n_gen=None, save_f=None):
        """
        Runs NEAT's genetic algorithm for at most n_gen generations.  If n is None, run until solution is found or
        extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign a Python float to the `fitness` member
        of each genome.

        The fitness function is free to maintain external state, perform evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes, the genomes themselves (apart from
        updating the fitness member), or the configuration object.
        
        :param calculate_fitness: Method used to determine the fitness function for each of the candidates
        :param eval_generation: Environment function used in the first stage
        :param n_gen: Number of generations over which the population will evolve
        :param save_f: Method used to save the population after each generation
        """
        n_gen = n_gen if n_gen else float('inf')
        
        k = 0
        while n_gen is None or k < n_gen:
            k += 1
            
            # Call init on reporters
            self.reporters.start_generation(self.generation)
            
            # Evaluate all genomes using the user-provided function
            self.evaluate(calculate_fitness=calculate_fitness, eval_generation=eval_generation)
            
            # Gather and report statistics
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config,
                                         self.population,
                                         self.species,
                                         best)
            
            # Track the most fit genome
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best
            
            # Check if fitness-threshold is reached
            if not self.config.no_fitness_termination:
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break
            
            self.evolve()
            
            # Call end of the generation for the reporters
            self.reporters.end_generation(self.config, self.population, self.species)
            
            # Increment the generation-count
            self.generation += 1
            
            # Save the population by calling the provided save-function
            if save_f:
                save_f()
        
        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config,
                                          self.generation,
                                          self.best_genome)
        
        # Run the best genome at the end of the evaluation
        return self.best_genome
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def create_blueprint(self, game_id: int, observation_list: list):
        """
        Create a blueprint for a given game, based on the game's blueprint and all the candidates their final
        observations in the game. The blueprints get saved based on the self.save_blueprint_path parameter, the
        population's current generation and the game's ID.
        
        :param game_id: ID specifying the game
        :param observation_list: List of game.close() dictionaries
        """
        game = Game(game_id=game_id,
                    rel_path=self.rel_game_path,
                    silent=True)
        game.get_blueprint()
        
        # Get all the final positions of the agents
        positions = [o[D_POS] for o in observation_list]
        dot_x = [p[0] / PTM for p in positions]
        dot_y = [p[1] / PTM for p in positions]
        plt.plot(dot_x, dot_y, 'ro')
        
        # Add target again to map
        plt.plot(0.5, AXIS_Y - 0.5, 'go')
        
        # Add title
        plt.title("Blueprint - Game {id:05d} - Generation {gen:05d}".format(id=game_id, gen=self.generation))
        
        # Save the blueprint
        plt.savefig('{bp}game{id:05d}_gen{gen:05d}'.format(bp=self.save_blueprint_path,
                                                           id=game_id,
                                                           gen=self.generation))
    
    def evaluate(self, calculate_fitness, eval_generation):
        """
        This method consists out of three parts:
          1) Run the genomes in the environment, which returns a dictionary with all the keys of the genomes with their
             obtained list of Game-object close() observations: { genome_key : [game.close()] }
          2) Create a blueprint of each of the games which reflects the overall performance of the generation
          3) Evaluate the candidates on their corresponding fitness, this consists out of two different parts
              a) Evaluate the candidate's fitness for each of the games individually, thus resolving in a list of floats
                 (fitness) for each candidate for each of the games
              b) Combine all the fitness-values of all the games for every individual candidate to get the candidate's
                 overall fitness score
        
        The fitness of a candidate is saved in the candidate's self, so nothing must be returned.
        
        :param calculate_fitness: Method used to determine the fitness function for each of the candidates
        :param eval_generation: Environment function used in the first stage
        """
        # 1) Run the genomes in the environment
        observation_dict, games_list = eval_generation(genomes=list(iteritems(self.population)), config=self.config)
        
        # 2) Create the blueprints
        if self.save_blueprint_path != '':
            genome_keys = list(observation_dict.keys())
            for i, g in enumerate(games_list):
                game_obs = [observation_dict[k][i] for k in genome_keys]
                self.create_blueprint(game_id=g, observation_list=game_obs)
        
        # 3) Evaluate the candidates
        calculate_fitness(genomes=list(iteritems(self.population)),
                          observations=observation_dict)  # TODO: Possible to add reporter to this?
    
    def evolve(self):
        """
        The evolution-process consists out of two phases:
          1) Reproduction
          2) Speciation
        
        This method manipulates the population itself, so nothing has to be returned
        """
        # Create the next generation from the current generation
        self.population = self.reproduction.reproduce(self.config,
                                                      self.species,
                                                      self.config.pop_size,
                                                      self.generation)
        
        # Check for complete extinction
        if not self.species.species:
            self.reporters.complete_extinction()
            
            # If requested by the user, create a completely new population, otherwise raise an exception
            if self.config.reset_on_extinction:
                self.population = self.reproduction.create_new(self.config.genome_type,
                                                               self.config.genome_config,
                                                               self.config.pop_size)
            else:
                raise CompleteExtinctionException()
        
        # Divide the new population into species
        self.species.speciate(self.config, self.population, self.generation)
    
    # ---------------------------------------------> FUNCTIONAL METHODS <--------------------------------------------- #
    
    def add_reporter(self, reporter):
        """
        Add a new reporter to the population's ReporterSet.
        """
        self.reporters.add(reporter)
    
    def remove_reporter(self, reporter):
        """
        Remove the given reporter from the population's ReporterSet.
        """
        self.reporters.remove(reporter)
    
    def set_config(self, cfg):
        """
        Update the config-file.
        """
        self.config = cfg
        stagnation = cfg.stagnation_type(cfg.stagnation_config,
                                         self.reporters)
        self.reproduction = cfg.reproduction_type(cfg.reproduction_config,
                                                  self.reporters,
                                                  stagnation)
