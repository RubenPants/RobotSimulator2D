"""
main.py

Run a single population on one or more of the provided functionalities.
"""
import argparse
import traceback

from population.population import Population

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--blueprint', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--genome', type=bool, default=True)
    parser.add_argument('--live', type=bool, default=False)
    args = parser.parse_args()
    
    pop = Population(
            name="test",
            # version=1,
            folder_name='test',
    )
    if not pop.best_genome: pop.best_genome = list(pop.population.values())[0]
    # pop.population[9] = pop.population[list(pop.population.keys())[12]]
    # pop.save()
    # net = pop.make_net(pop.best_genome, pop.config, 1)
    # inp = pop.query_net(net, [[0] * 8])
    # print(inp)
    # raise Exception
    # pop.load(gen=1)
    
    try:
        if args.train:
            pop.log("\n===> TRAINING <===\n")
            from environment.training_env import TrainingEnv
            
            # Train for 100 generations
            trainer = TrainingEnv()
            trainer.evaluate_and_evolve(
                    pop,
                    n=args.iterations,
                    # parallel=False,
            )
        
        if args.blueprint:
            pop.log("\n===> CREATING BLUEPRINTS <===\n")
            from environment.training_env import TrainingEnv
            
            # Create the blueprints for first 5 games
            trainer = TrainingEnv()
            for g in range(1, 6):
                pop.log(f"Creating blueprints for  game {g}")
                trainer.set_games([g])
                # for i in range(11):
                #     pop.load(gen=int(i * 10))
                #     trainer.blueprint_genomes(pop)
                trainer.blueprint_genomes(pop)
        
        if args.evaluate:
            pop.log("\n===> EVALUATING <===\n")
            from environment.evaluation_env import EvaluationEnv
            
            evaluator = EvaluationEnv()
            evaluator.evaluate_genome_list(
                    genome_list=[pop.best_genome],
                    pop=pop,
            )
        
        if args.genome:
            pop.log("\n===> VISUALIZING GENOME <===\n")
            # genome = list(pop.population.values())[2]
            genome = pop.best_genome
            pop.log(f"Genome size: {genome.size()}")
            pop.visualize_genome(
                    debug=True,
                    genome=genome,
            )
        
        if args.live:
            print("\n===> STARTING LIVE DEMO <===\n")
            from environment.visualizer import Visualizer
            
            net = pop.make_net(pop.best_genome, pop.config, 1)
            visualizer = Visualizer(
                    query_net=pop.query_net,
                    debug=False,
                    # speedup=1,
            )
            
            visualizer.visualize(
                    network=net,
                    game_id=1,
            )
    except Exception as e:
        pop.log(traceback.format_exc())
        raise e
