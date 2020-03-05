"""
run_populations.py

Run a sequence of populations for each a different configuration.
"""
import argparse

from configs.config import NeatConfig
from population.population import Population
from utils.dictionary import *


def main(folder,
         fitness,
         gru,
         reproduce,
         train=False,
         train_iterations=0,
         blueprint=False,
         evaluate=False,
         ):
    """
    Run a population's configuration.
    
    :param folder: Folder to which the population is stored (NEAT, NEAT-GRU, ...)
    :param fitness: Fitness function used to evaluate the population
    :param gru: Enable GRU-mutations in the population
    :param reproduce: Have sexual reproduction
    :param train: Train the population
    :param train_iterations: Number of training generations
    :param blueprint: Create a blueprint evaluation for the population
    :param evaluate: Evaluate the best genome of the population
    """
    print(f"\n----------> RUNNING FOR THE {folder} - {fitness} CONFIGURATION <----------")
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
        trainer.evaluate_and_evolve(pop, n=train_iterations)
    
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
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--blueprint', type=bool, default=True)
    parser.add_argument('--evaluate', type=bool, default=False)
    args = parser.parse_args()
    
    for reproduce in [True, False]:
        for gru in [True, False]:
            folder = 'NEAT-GRU' if gru else 'NEAT'
            for fitness in D_FIT_OPTIONS:
                main(
                        folder=folder,
                        fitness=fitness,
                        gru=gru,
                        reproduce=reproduce,
                        train=args.train,
                        train_iterations=args.iterations,
                        blueprint=args.blueprint,
                        evaluate=args.evaluate,
                
                )
