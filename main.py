import argparse

from population.population import Population
from pytorch_neat.recurrent_net import RecurrentNet


def make_net(genome, config, bs):
    """
    Create the "brains" of the candidate, based on its genetic wiring.

    :param genome: Genome specifies the brains internals
    :param config: Configuration class
    :param bs: Batch size, which represents amount of games trained in parallel
    """
    return RecurrentNet.create(genome, config, bs)


def query_net(net, states):
    """
    Call the net (brain) to determine the best suited action, given the stats (current environment observation)

    :param net: RecurrentNet, created in 'make_net' (previous method)
    :param states: Current observations for each of the games
    """
    outputs = net.activate(states).numpy()
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--blueprint', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--genome', type=bool, default=False)
    parser.add_argument('--live', type=bool, default=True)
    args = parser.parse_args()
    
    pop = Population(
            name="test",
            version=1,
            make_net_method=make_net,
            query_net_method=query_net,
    )
    # pop.load(gen=1)
    
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
        pop.visualize_genome(
                debug=True,
        )
    
    if args.live:
        print("\n===> STARTING LIVE DEMO <===\n")
        from environment.visualizer import Visualizer
        
        net = make_net(pop.best_genome, pop.config, 1)
        visualizer = Visualizer(
                query_net=query_net,
                debug=False,
                # speedup=1,
        )
        
        visualizer.visualize(
                network=net,
                game_id=2,
        )
