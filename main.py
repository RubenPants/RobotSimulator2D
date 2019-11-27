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
    pop = Population(
            name='distance',
            rel_path='control/NEAT/',
            make_net_method=make_net,
            query_net_method=query_net,
    )
    
    # """
    # Evaluation
    from environment.evaluator import Evaluator
    
    evaluator = Evaluator(
            rel_path='environment/',
    )
    
    # Train for 100 generations
    # evaluator.evaluate_and_evolve(pop, n=100)
    # evaluator.set_games([1])
    for i in range(11):
        pop.load(gen=int(i * 10))
        evaluator.blueprint_genomes(pop)
    
    """
    
    # Visualization
    from environment.visualizer import Visualizer
    
    net = make_net(pop.best_genome, pop.config, 1)
    visualizer = Visualizer(
            query_net=query_net,
            rel_path='environment/',
            debug=False,
    )
    
    visualizer.visualize(net, 6)
    
    # """
