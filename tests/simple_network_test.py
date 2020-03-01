"""
network_test.py

Test the possible network configurations.
"""
import os
import unittest
from random import random

import torch

from configs.config import NeatConfig
from population.population import query_net
from population.utils.config.default_config import Config
from population.utils.genome_util.genome import DefaultGenome
from population.utils.network_util.activations import tanh_activation
from population.utils.network_util.feed_forward_net import make_net
from population.utils.population_util.reproduction import DefaultReproduction
from population.utils.population_util.species import DefaultSpeciesSet
from population.utils.population_util.stagnation import DefaultStagnation
from utils.dictionary import *


def get_genome(inputs, hidden, outputs):
    """Create a simple feedforward neuron."""  # Get the configuration
    cfg = NeatConfig()
    cfg.num_inputs = inputs
    cfg.num_hidden = hidden
    cfg.num_outputs = outputs
    cfg.initial_connection = D_FULL_DIRECT
    cfg.enable_gru = False  # Only simple hidden nodes allowed
    config = Config(
            genome_type=DefaultGenome,
            reproduction_type=DefaultReproduction,
            species_set_type=DefaultSpeciesSet,
            stagnation_type=DefaultStagnation,
            config=cfg,
    )
    
    # Create the genome
    g = DefaultGenome(key=0)
    g.configure_new(config.genome_config)
    return g, config


class TestFeedForward(unittest.TestCase):
    def test_simple_network(self, debug=True):
        """
        Test the functionality of a simple network with only one input, one output, and no hidden nodes.
        Bias will be put to zero.
        
        Network:
            I -- O
        """
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the genome and its corresponding config file
        genome, config = get_genome(1, 0, 1)
        
        # Manipulate the genome's biases and connection weights
        genome.nodes[0].bias = 0  # Output-bias
        genome.connections[(-1, 0)].weight = 1  # Single connection from input to output
        
        # Change the output's activation
        genome.nodes[0].activation = D_LINEAR
        if debug: print(genome)
        
        # Create a network
        net = make_net(genome, config, bs=1)
        
        # Query the network; each input is directly mapped on the output (under tanh activation function)
        for _ in range(100):
            r = random()
            o = query_net(net, [[r]])
            assert float(tanh_activation(torch.tensor(r, dtype=torch.float64))) == o


if __name__ == '__main__':
    unittest.main()
