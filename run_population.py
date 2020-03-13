"""
run_populations.py

Run a sequence of populations for each a different configuration.
"""
import argparse

from config import NeatConfig
from population.population import Population
from utils.dictionary import *


def main(fitness,
         gru,
         reproduce,
         train=False,
         train_iterations=0,
         blueprint=False,
         evaluate=False,
         ):
    """
    Run a population's configuration.
    
    :param fitness: Fitness function used to evaluate the population
    :param gru: Enable GRU-mutations in the population
    :param reproduce: Have sexual reproduction
    :param train: Train the population
    :param train_iterations: Number of training generations
    :param blueprint: Create a blueprint evaluation for the population
    :param evaluate: Evaluate the best genome of the population
    """
    folder = D_NEAT_GRU if gru else D_NEAT
    
    # Give overview of population
    print(f"\n===> RUNNING FOR THE FOLLOWING CONFIGURATION: <===")
    print(f"\t> fitness: {fitness}")
    print(f"\t> enable_gru: {gru}")
    print(f"\t> sexual_reproduction: {reproduce}")
    print(f"\t> Saving under folder: {folder}")
    print(f"\t> Train: {train} ({train_iterations} iterations)")
    print(f"\t> Create blueprints: {blueprint}")
    print()
    
    # Modify configuration correspondingly and create the population
    cfg = NeatConfig()
    cfg.fitness = fitness
    cfg.enable_gru = gru
    cfg.sexual_reproduction = reproduce
    pop = Population(
            version=1,
            config=cfg,
            folder_name=folder,
    )
    
    if train:
        print("\n===> TRAINING <===\n")
        from environment.training_env import TrainingEnv
        
        # Train for 100 generations
        trainer = TrainingEnv()
        trainer.evaluate_and_evolve(
                pop,
                n=train_iterations,
                # parallel=False,
        )
    
    if blueprint:
        print("\n===> CREATING BLUEPRINTS <===\n")
        from environment.training_env import TrainingEnv
        
        # Create the blueprints for first 5 games
        trainer = TrainingEnv()
        for g in range(1, 6):
            print(f"Creating blueprints for  game {g}")
            trainer.set_games([g])
            trainer.blueprint_genomes(pop)
    
    if evaluate:
        print("\n===> EVALUATING <===\n")
        from environment.evaluation_env import EvaluationEnv
        
        evaluator = EvaluationEnv()
        evaluator.evaluate_genome_list(
                genome_list=[pop.best_genome],
                pop=pop,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=0)
    parser.add_argument('--reproduce', type=int, default=0)
    parser.add_argument('--enable_gru', type=int, default=0)
    parser.add_argument('--fitness', type=str, default='')
    parser.add_argument('--blueprint', type=int, default=0)
    parser.add_argument('--evaluate', type=int, default=0)
    args = parser.parse_args()
    
    main(
            fitness=args.fitness,
            gru=bool(args.enable_gru),
            reproduce=bool(args.reproduce),
            train=bool(args.train),
            train_iterations=args.iterations,
            blueprint=bool(args.blueprint),
            evaluate=bool(args.evaluate),
    )
