"""
main.py

Run a single population on one or more of the provided functionalities.
"""
import argparse
import traceback

from population.population import Population

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    # Main methods
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--blueprint', type=bool, default=False)
    parser.add_argument('--trace', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--genome', type=bool, default=False)
    parser.add_argument('--live', type=bool, default=True)
    
    # Extra arguments
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--unused_cpu', type=int, default=2)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    
    pop = Population(
            name='test',
            # version=1,
            folder_name='test',
    )
    if not pop.best_genome: pop.best_genome = list(pop.population.values())[0]
    # pop.best_genome = list(pop.population.values())[3]  # TODO
    # pop.population = {k: v for k, v in pop.population.items() if k in [1033]}  # TODO
    
    try:
        if args.train:
            pop.log("\n===> TRAINING <===\n")
            from environment.env_training import TrainingEnv
            
            # Train for 100 generations
            trainer = TrainingEnv(
                    unused_cpu=args.unused_cpu,  # Use two cores less to keep laptop usable
                    game_config=pop.game_config,
            )
            trainer.evaluate_and_evolve(
                    pop,
                    n=args.iterations,
                    parallel=not args.debug,
            )
        
        if args.blueprint:
            pop.log("\n===> CREATING BLUEPRINTS <===\n")
            from environment.env_visualizing import VisualizingEnv
            
            # Create the blueprints for first 5 games
            visualizer = VisualizingEnv(game_config=pop.game_config)
            games = [g for g in range(1, 6)]
            pop.log(f"Creating blueprints for games: {games}")
            visualizer.set_games(games)
            visualizer.blueprint_genomes(pop)
        
        if args.trace:
            pop.log("\n===> CREATING TRACES <===\n")
            from environment.env_visualizing import VisualizingEnv
            
            # Create the traces for first 5 games
            visualizer = VisualizingEnv(game_config=pop.game_config)
            games = [g for g in range(1, 6)]
            pop.log(f"Creating traces for games: {games}")
            visualizer.set_games(games)
            visualizer.trace_genomes(pop)
        
        if args.evaluate:
            pop.log("\n===> EVALUATING <===\n")
            from environment.env_evaluation import EvaluationEnv
            
            evaluator = EvaluationEnv(game_config=pop.game_config)
            evaluator.evaluate_genome_list(
                    genome_list=[pop.best_genome],
                    pop=pop,
            )
        
        if args.genome:
            print("\n===> VISUALIZING GENOME <===\n")
            # genome = list(pop.population.values())[2]
            genome = pop.best_genome
            print(f"Genome {genome.key} with size: {genome.size()}")
            pop.visualize_genome(
                    # debug=args.debug,  TODO
                    debug=True,
                    genome=genome,
            )
        
        if args.live:
            print("\n===> STARTING LIVE DEMO <===\n")
            from environment.env_visualizing_live import LiveVisualizer

            genome = pop.best_genome
            print(f"Genome {genome.key} with size: {genome.size()}")
            net = pop.make_net(genome=genome, config=pop.config, game_config=pop.game_config, bs=1)
            visualizer = LiveVisualizer(
                    query_net=pop.query_net,
                    debug=args.debug,
                    game_config=pop.game_config,
                    # speedup=1,
            )
            
            visualizer.visualize(
                    network=net,
                    game_id=1,
            )
    except Exception as e:
        pop.log(traceback.format_exc(), print_result=False)
        raise e
