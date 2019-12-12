import argparse

from control.entities.population import Population
from pytorch_neat.recurrent_net import RecurrentNet


def make_net(genome, config, bs):
    """
    Create the "brains" of the candidate, based on its genetic wiring.

    :param genome: Genome specifies the brains internals
    :param config: Configuration file from this folder: config.cfg
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
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()
    
    pop = Population(
            name='path_time',
            rel_path='control/NEAT/',
            make_net_method=make_net,
            query_net_method=query_net,
    )
    if args.train:
        print("\n===> TRAINING <===\n")
        from environment.training_env import TrainingEnv
        
        trainer = TrainingEnv(
                rel_path='environment/',
        )
        
        # Train for 100 generations
        trainer.evaluate_and_evolve(pop, n=100)
        
        # Create the blueprints for first 5 games
        for g in range(1, 6):
            print("Evaluate on game {}".format(g))
            trainer.set_games([g])
            for i in range(11):
                pop.load(gen=int(i * 10))
                trainer.blueprint_genomes(pop)
    
    if args.evaluate:
        print("\n===> EVALUATING <===\n")
        from environment.evaluation_env import EvaluationEnv
        
        evaluator = EvaluationEnv(
                rel_path='environment/',
        )
        evaluator.evaluate_genome_list(genome_list=[pop.best_genome], pop=pop)
        
    if args.visualize:
        print("\n===> VISUALIZING <===\n")
        pop.visualize_genome()
        
        from environment.visualizer import Visualizer
        
        net = make_net(pop.best_genome, pop.config, 1)
        visualizer = Visualizer(
                query_net=query_net,
                rel_path='environment/',
                debug=False,
                speedup=1,
        )
        
        visualizer.visualize(net, 1)
