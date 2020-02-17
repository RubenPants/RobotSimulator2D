"""
main.py

TODO
"""
import os
from control.entities.population_manager import PopulationManager

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


def main(name='test', rel_path='', silent=False):
    # Load in or create the population
    pop = PopulationManager(make_net_method=make_net,
                            name=name,
                            query_net_method=query_net,
                            rel_path=rel_path,
                            silent=silent)
    pop.evaluate_random_mazes(n_generations=1)
    # pop.evaluate_chosen_maze(n_generations=20, maze_id=0)
    # pop.visualize(game_id=0, speedup=5, debug=False)


if __name__ == "__main__":
    # Go back to root
    os.chdir("../..")
    
    # Call main
    main()
