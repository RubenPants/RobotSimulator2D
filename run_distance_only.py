"""
run_distance_only.py

Train a single population on the distance-only task, which will always train first for a requested set of iterations
on the train_same_games task, after which the traces of the best genome on each of the five games is determined.
"""
import argparse
import traceback

from config import Config
from main import trace_most_fit, train_same_games
from population.population import Population
from process_killer import main as process_killer
from utils.dictionary import *


def main(gru,
         reproduce,
         train_iterations=0,
         version=0,
         ):
    """
    Run a population's configuration.

    :param gru: Enable GRU-mutations in the population
    :param reproduce: Have sexual reproduction
    :param train_iterations: Number of training generations
    :param version: Version of the model
    """
    # Set the fixed configs
    folder = D_DISTANCE_ONLY
    config = Config()
    config.evaluation.fitness = D_DISTANCE  # Always use the distance-fitness
    config.species.stagnation = 25  # Greater since improvement comes slow
    config.game.duration = 50  # Limited time to find target, but just enough for fastest genomes
    
    # Let inputs apply to configuration
    config.genome.gru_enabled = gru
    config.reproduction.sexual = reproduce
    
    # Create the population
    pop = Population(
            version=version,
            config=config,
            folder_name=folder,
    )
    
    # Give overview of population
    msg = f"\n===> RUNNING DISTANCE-ONLY FOR THE FOLLOWING CONFIGURATION: <===" \
          f"\n\t> gru_enabled: {config.genome.gru_enabled}" \
          f"\n\t> sexual_reproduction: {config.reproduction.sexual}" \
          f"\n\t> Train for {train_iterations} iterations\n"
    pop.log(msg)
    
    # Set games used for evaluation
    games = [99995, 99996, 99997, 99998, 99999]
    
    # Execute the segments
    try:
        train_same_games(
                games=games,
                game_config=config,
                population=pop,
                unused_cpu=0,
                iterations=train_iterations,
                debug=False,
        )
        
        trace_most_fit(
                population=pop,
                game_config=config,
                genome=pop.best_genome,
                games=games,
        )
    
    except Exception as e:
        pop.log(traceback.format_exc(), print_result=False)
        raise e
    finally:
        process_killer('run_distance_only.py')  # Close all the terminated files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gru_enabled', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=0)
    parser.add_argument('--reproduce', type=int, default=0)
    parser.add_argument('--version', type=int, default=0)
    args = parser.parse_args()
    
    main(
            gru=bool(args.gru_enabled),
            reproduce=bool(args.reproduce),
            train_iterations=args.iterations,
            version=args.version,
    )
