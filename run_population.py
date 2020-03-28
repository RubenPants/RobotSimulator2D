"""
run_populations.py

Run a sequence of populations for each a different configuration.
"""
import argparse
import traceback

from config import Config
from configs.bot_config import get_proximity_angles
from main import blueprint, evaluate, trace, trace_most_fit, train, visualize_genome
from population.population import Population
from process_killer import main as process_killer
from utils.dictionary import *


def main(fitness,
         gru,
         reproduce,
         train_iterations=0,
         version=0,
         ):
    """
    Run a population's configuration.
    
    :param fitness: Fitness function used to evaluate the population
    :param gru: Enable GRU-mutations in the population
    :param reproduce: Have sexual reproduction
    :param train_iterations: Number of training generations
    :param version: Version of the model
    """
    # Re-configure the config-file
    folder = D_NEAT_GRU if gru else D_NEAT
    cfg = Config()
    cfg.bot.angular_dir = [True, False]
    cfg.bot.delta_dist_enabled = False
    cfg.bot.prox_angles = get_proximity_angles()
    cfg.game.duration = 60  # 60 seconds should be enough to reach each of the targets
    cfg.game.fps = 20
    cfg.population.pop_size = 256
    cfg.population.compatibility_thr = 3.0
    
    # Let inputs apply to configuration
    cfg.genome.gru_enabled = gru
    cfg.evaluation.fitness = fitness
    cfg.population.crossover_enabled = reproduce
    cfg.update()
    
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
          f"\n\t> sexual_reproduction: {cfg.population.crossover_enabled}" \
          f"\n\t> Saving under folder: {folder}" \
          f"\n\t> Training for {train_iterations} iterations\n"
    pop.log(msg)
    
    # Set games used for evaluation
    games = [g for g in range(1, 6)]
    
    # Execute the requested segments
    try:
        train(
                population=pop,
                game_config=cfg,
                unused_cpu=0,
                iterations=train_iterations,
                debug=False,
        )
        
        # Evaluate the trained population
        blueprint(
                population=pop,
                game_config=cfg,
                games=games,
        )
        trace(
                population=pop,
                game_config=cfg,
                games=games,
        )
        trace_most_fit(
                population=pop,
                game_config=cfg,
                genome=pop.best_genome,
                games=games,
        )
        evaluate(
                population=pop,
                game_config=cfg,
        )
        visualize_genome(
                population=pop,
                genome=pop.best_genome,
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
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--version', type=int, default=0)
    args = parser.parse_args()
    
    main(
            fitness=args.fitness,
            gru=bool(args.gru_enabled),
            reproduce=bool(args.reproduce),
            train_iterations=args.iterations,
            version=args.version,
    )
