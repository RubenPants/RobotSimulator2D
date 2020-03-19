"""
run_populations.py

Run a sequence of populations for each a different configuration.
"""
import argparse
import traceback

from config import NeatConfig
from population.population import Population
from utils.dictionary import *


def main(fitness,
         gru,
         reproduce,
         blueprint=False,
         evaluate=False,
         trace=False,
         train=False,
         train_iterations=0,
         version=0,
         ):
    """
    Run a population's configuration.
    
    :param fitness: Fitness function used to evaluate the population
    :param gru: Enable GRU-mutations in the population
    :param reproduce: Have sexual reproduction
    :param blueprint: Create a blueprint evaluation for the population for the first 5 games
    :param evaluate: Evaluate the best genome of the population
    :param trace: Create a trace evaluation for the population for the first 5 games
    :param train: Train the population
    :param train_iterations: Number of training generations
    :param version: Version of the model
    """
    # Let inputs apply to configuration
    folder = D_NEAT_GRU if gru else D_NEAT
    cfg = NeatConfig()
    cfg.fitness = fitness
    cfg.gru_enabled = gru
    cfg.sexual_reproduction = reproduce
    
    # Create the population
    pop = Population(
            version=version,
            neat_config=cfg,
            folder_name=folder,
    )
    
    # Give overview of population
    msg = f"\n===> RUNNING FOR THE FOLLOWING CONFIGURATION: <===" \
          f"\n\t> fitness: {cfg.fitness}" \
          f"\n\t> gru_enabled: {cfg.gru_enabled}" \
          f"\n\t> sexual_reproduction: {cfg.sexual_reproduction}" \
          f"\n\t> Saving under folder: {folder}" \
          f"\n\t> Train: {train} ({train_iterations} iterations)" \
          f"\n\t> Create blueprints: {blueprint}" \
          f"\n\t> Create traces: {trace}\n"
    pop.log(msg)
    
    try:
        if train:
            pop.log("\n===> TRAINING <===\n")
            from environment.env_training import TrainingEnv
            
            # Train for 100 generations
            trainer = TrainingEnv(game_config=pop.game_config)
            trainer.evaluate_and_evolve(
                    pop,
                    n=train_iterations,
                    # parallel=False,
            )
        
        if blueprint:
            pop.log("\n===> CREATING BLUEPRINTS <===\n")
            from environment.env_visualizing import VisualizingEnv
            
            # Create the blueprints for first 5 games
            visualizer = VisualizingEnv(game_config=pop.game_config)
            games = [g for g in range(1, 6)]
            pop.log(f"Creating blueprints for games: {games}")
            visualizer.set_games(games)
            visualizer.blueprint_genomes(pop)
        
        if trace:
            pop.log("\n===> CREATING TRACES <===\n")
            from environment.env_visualizing import VisualizingEnv
            
            # Create the blueprints for first 5 games
            visualizer = VisualizingEnv(game_config=pop.game_config)
            games = [g for g in range(1, 6)]
            pop.log(f"Creating traces for games: {games}")
            visualizer.set_games(games)
            visualizer.trace_genomes(pop)
        
        if evaluate:
            pop.log("\n===> EVALUATING <===\n")
            from environment.env_evaluation import EvaluationEnv
            
            evaluator = EvaluationEnv(game_config=pop.game_config)
            evaluator.evaluate_genome_list(
                    genome_list=[pop.best_genome],
                    pop=pop,
            )
    except Exception as e:
        pop.log(traceback.format_exc(), print_result=False)
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--fitness', type=str, default='')
    parser.add_argument('--gru_enabled', type=int, default=1)
    parser.add_argument('--reproduce', type=int, default=1)
    parser.add_argument('--blueprint', type=int, default=1)
    parser.add_argument('--evaluate', type=int, default=0)
    parser.add_argument('--trace', type=int, default=0)
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=0)
    parser.add_argument('--version', type=int, default=2)
    args = parser.parse_args()
    
    main(
            fitness=args.fitness,
            gru=bool(args.gru_enabled),
            reproduce=bool(args.reproduce),
            blueprint=bool(args.blueprint),
            evaluate=bool(args.evaluate),
            trace=bool(args.trace),
            train=bool(args.train),
            train_iterations=args.iterations,
            version=args.version,
    )
