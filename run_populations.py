"""
run_populations.py

Run a sequence of populations for each a different configuration.
"""
import argparse

from configs.config import NeatConfig
from population.population import Population
from utils.dictionary import D_FIT_OPTIONS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--blueprint', type=bool, default=True)
    parser.add_argument('--evaluate', type=bool, default=False)
    args = parser.parse_args()
    
    for fitness in D_FIT_OPTIONS:
        print(f"\n----------> RUNNING FOR THE {fitness} CONFIGURATION <----------")
        cfg = NeatConfig()
        cfg.fitness = fitness
        pop = Population(
                version=1,
                config=cfg,
        )
        
        if args.train:
            print("\n===> TRAINING <===\n")
            from environment.training_env import TrainingEnv
            
            # Train for 100 generations
            trainer = TrainingEnv()
            trainer.evaluate_and_evolve(pop, n=args.iterations)
        
        if args.blueprint:
            print("\n===> CREATING BLUEPRINTS <===\n")
            from environment.training_env import TrainingEnv
            
            # Create the blueprints for first 5 games
            trainer = TrainingEnv()
            for g in range(1, 6):
                print(f"Creating blueprints for  game {g}")
                trainer.set_games([g])
                trainer.blueprint_genomes(pop)
        
        if args.evaluate:
            print("\n===> EVALUATING <===\n")
            from environment.evaluation_env import EvaluationEnv
            
            evaluator = EvaluationEnv()
            evaluator.evaluate_genome_list(
                    genome_list=[pop.best_genome],
                    pop=pop,
            )
