"""
run_populations.py

Run a sequence of populations for each a different configuration.
"""
import argparse
import traceback

from config import Config
from main import blueprint, evaluate, trace, train
from population.population import Population
from process_killer import main as process_killer
from utils.dictionary import *


def main(fitness,
         gru,
         reproduce,
         run_blueprint=False,
         run_evaluate=False,
         run_trace=False,
         run_train=False,
         train_iterations=0,
         version=0,
         ):
    """
    Run a population's configuration.
    
    :param fitness: Fitness function used to evaluate the population
    :param gru: Enable GRU-mutations in the population
    :param reproduce: Have sexual reproduction
    :param run_blueprint: Create a blueprint evaluation for the population for the first 5 games
    :param run_evaluate: Evaluate the best genome of the population
    :param run_trace: Create a trace evaluation for the population for the first 5 games
    :param run_train: Train the population
    :param train_iterations: Number of training generations
    :param version: Version of the model
    """
    # Let inputs apply to configuration
    folder = D_NEAT_GRU if gru else D_NEAT
    cfg = Config()
    cfg.evaluation.fitness = fitness
    cfg.genome.gru_enabled = gru
    cfg.reproduction.sexual = reproduce
    
    # Create the population
    pop = Population(
            version=version,
            config=cfg,
            folder_name=folder,
    )
    
    # Give overview of population
    msg = f"\n===> RUNNING FOR THE FOLLOWING CONFIGURATION: <===" \
          f"\n\t> fitness: {cfg.evaluation.fitness}" \
          f"\n\t> gru_enabled: {cfg.genome.gru_enabled}" \
          f"\n\t> sexual_reproduction: {cfg.reproduction.sexual}" \
          f"\n\t> Saving under folder: {folder}" \
          f"\n\t> Train: {run_train} ({train_iterations} iterations)" \
          f"\n\t> Create blueprints: {run_blueprint}" \
          f"\n\t> Create traces: {run_trace}\n"
    pop.log(msg)
    
    # Set games used for evaluation
    games = [g for g in range(1, 6)]
    
    # Execute the requested segments
    try:
        if run_train:
            train(
                    population=pop,
                    game_config=cfg,
                    unused_cpu=0,
                    iterations=train_iterations,
                    debug=False,
            )
        
        if run_blueprint:
            blueprint(
                    population=pop,
                    game_config=cfg,
                    games=games,
            )
        
        if run_trace:
            trace(
                    population=pop,
                    game_config=cfg,
                    games=games,
            )
        
        if run_evaluate:
            evaluate(
                    population=pop,
                    game_config=cfg,
            )
    
    except Exception as e:
        pop.log(traceback.format_exc(), print_result=False)
        raise e
    finally:
        process_killer('run_population.py')  # Close all the terminated files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--fitness', type=str, default=D_DISTANCE)
    parser.add_argument('--gru_enabled', type=int, default=1)
    parser.add_argument('--reproduce', type=int, default=1)
    parser.add_argument('--blueprint', type=int, default=0)
    parser.add_argument('--evaluate', type=int, default=0)
    parser.add_argument('--trace', type=int, default=0)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--version', type=int, default=0)
    args = parser.parse_args()
    
    main(
            fitness=args.fitness,
            gru=bool(args.gru_enabled),
            reproduce=bool(args.reproduce),
            run_blueprint=bool(args.blueprint),
            run_evaluate=bool(args.evaluate),
            run_trace=bool(args.trace),
            run_train=bool(args.train),
            train_iterations=args.iterations,
            version=args.version,
    )
