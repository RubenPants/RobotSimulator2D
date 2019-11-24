import os

import neat

from control.entities.population import Population
from environment.evaluator import Evaluator
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
    config_path = os.path.join(os.path.dirname(__file__), "control/NEAT/config.cfg")
    config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
    )
    
    pop = Population(
            name='test',
            rel_path='',
            make_net_method=make_net,
            query_net_method=query_net,
            config=config
    )
    
    evaluator = Evaluator(
            rel_path='environment/'
    )
    
    evaluator.single_evaluation(pop)
