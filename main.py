import argparse

from population.population import Population

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--blueprint', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--genome', type=bool, default=True)
    parser.add_argument('--live', type=bool, default=False)
    args = parser.parse_args()
    
    pop = Population(
            name="test",
            # version=1,
    )
    # pop.population[9] = pop.population[list(pop.population.keys())[12]]
    # pop.save()
    # net = pop.make_net(pop.best_genome, pop.config, 1)
    # inp = pop.query_net(net, [[0] * 8])
    # print(inp)
    # raise Exception
    # pop.load(gen=1)
    
    if args.train:
        print("\n===> TRAINING <===\n")
        from environment.training_env import TrainingEnv
        
        # Train for 100 generations
        trainer = TrainingEnv()
        trainer.evaluate_and_evolve(
                pop,
                n=args.iterations,
                parallel=False,
        )
    
    if args.blueprint:
        print("\n===> CREATING BLUEPRINTS <===\n")
        from environment.training_env import TrainingEnv
        
        # Create the blueprints for first 5 games
        trainer = TrainingEnv()
        for g in range(1, 6):
            print(f"Creating blueprints for  game {g}")
            trainer.set_games([g])
            # for i in range(11):
            #     pop.load(gen=int(i * 10))
            #     trainer.blueprint_genomes(pop)
            trainer.blueprint_genomes(pop)
    
    if args.evaluate:
        print("\n===> EVALUATING <===\n")
        from environment.evaluation_env import EvaluationEnv
        
        evaluator = EvaluationEnv()
        evaluator.evaluate_genome_list(
                genome_list=[pop.best_genome],
                pop=pop,
        )
    
    if args.genome:
        print("\n===> VISUALIZING GENOME <===\n")
        genome = list(pop.population.values())[3]
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
