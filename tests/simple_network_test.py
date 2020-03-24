"""
simple_network_test.py

Test the possible network configurations for the simple networks (i.e. feedforward network with simple hidden nodes).
"""
import os
import unittest
from random import random

import torch

from config import GameConfig, NeatConfig
from population.population import query_net
from population.utils.config.default_config import Config
from population.utils.genome_util.genome import DefaultGenome
from population.utils.network_util.activations import relu_activation, tanh_activation
from population.utils.network_util.feed_forward_net import make_net
from population.utils.population_util.reproduction import DefaultReproduction
from population.utils.population_util.species import DefaultSpecies
from population.utils.population_util.stagnation import DefaultStagnation
from utils.dictionary import *

# Precision
EPSILON = 1e-5


def get_genome(outputs):
    """Create a simple feedforward neuron."""
    # Get the configuration
    cfg = NeatConfig()
    cfg.num_outputs = outputs
    cfg.initial_connection = D_FULL_NODIRECT  # input -> hidden -> output
    cfg.gru_enabled = False  # Only simple hidden nodes allowed
    config = Config(
            genome_type=DefaultGenome,
            reproduction_type=DefaultReproduction,
            species_type=DefaultSpecies,
            stagnation_type=DefaultStagnation,
            config=cfg,
    )
    
    # Create the genome
    g = DefaultGenome(key=0, num_outputs=cfg.num_outputs)
    g.configure_new(config.genome_config)
    return g, config


class TestFeedForward(unittest.TestCase):
    def test_1inp_1out(self, debug=False):
        """> Test single feedforward network with one input and one output.
        
        :note: Bias will be put to zero, and connection weights to 1.
        
        In this test, the value of the input will be mapped directly onto the output, where it will be squished by the
        output's squishing function (tanh).
        
        Network:
            I -- O  ==  (-1) -- (0)
        """
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the genome and its corresponding config file
        genome, config = get_genome(1)
        
        # Manipulate the genome's connections and biases
        genome.connections = dict()
        for i, o in [(-1, 0)]:
            genome.create_connection(config=config.genome_config, input_key=i, output_key=o, weight=1.0)
        genome.nodes[0].bias = 0  # Output-bias
        if debug: print(genome)
        
        # Create a network
        game_config = GameConfig()
        net = make_net(genome=genome, config=config, game_config=game_config, bs=1)
        
        # Query the network; each input is directly mapped on the output (under tanh activation function)
        for _ in range(100):
            r = random() * 2 - 1
            self.assertAlmostEqual(float(tanh_activation(torch.tensor(r, dtype=torch.float64))),
                                   query_net(net, [[r]]),
                                   delta=EPSILON)
    
    def test_1inp_1hid_1out(self, debug=False):
        """> Test single feedforward network with one input, one hidden node, and one output.
        
        :note: Bias will be put to zero, and connection weights to 1.
        
        In this test, the value of the input progresses in two steps to reach the output:
          1) input-to-hidden: The value of the input node squished by the hidden node's relu function
          2) hidden-to-output: The value of the hidden node squished by the output node's tanh function
        This flow is executed in one go, since each iteration, all hidden nodes are updated before taking the output
        nodes into account.
        
        :note: The relu makes every negative input equal to zero. Positive inputs will be simply forwarded.

        Network:
            I -- H -- O  ==  (-1) -- (1) -- (0)
        """
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the genome and its corresponding config file
        genome, config = get_genome(1)
        genome.nodes[1] = genome.create_node(config=config.genome_config, node_id=1)
        
        # Manipulate the genome's biases and connection weights
        genome.nodes[0].bias = 0  # Output-bias
        genome.nodes[1].bias = 0  # Hidden-bias
        genome.connections = dict()
        for i, o in [(-1, 1), (1, 0)]:
            genome.create_connection(config=config.genome_config, input_key=i, output_key=o, weight=1.0)
        if debug: print(genome)
        
        # Create a network
        game_config = GameConfig()
        net = make_net(genome=genome, config=config, game_config=game_config, bs=1)
        
        # Query the network; single input in range [-1, 1]
        inputs = [random() * 2 - 1 for _ in range(100)]
        hidden_values = [relu_activation(torch.tensor(i, dtype=torch.float64)) for i in inputs]
        output_values = [tanh_activation(h) for h in hidden_values]
        
        # Query the network for the values
        for idx, inp in enumerate(inputs):
            [[r]] = query_net(net, [[inp]])
            self.assertAlmostEqual(r,
                                   float(output_values[idx]),
                                   delta=EPSILON)
    
    def test_1inp_2hid_1out(self, debug=False):
        """> Test single feedforward network with one input, two hidden nodes, and one output.
        
        :note: Bias will be put to zero, and connection weights to 1.
        
        For this test, in contrast to 'test_1inp_1hid_1out', the outputs aren't directly mapped from the inputs, but
        are delayed with one time-step (due to the double hidden nodes).

        Network:
            I -- H -- H -- O  ==  (-1) -- (1) -- (2) -- (0)
        """
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the genome and its corresponding config file
        genome, config = get_genome(1)
        genome.nodes[1] = genome.create_node(config=config.genome_config, node_id=1)
        genome.nodes[2] = genome.create_node(config=config.genome_config, node_id=2)
        
        # Manipulate the genome's biases and connection weights
        genome.nodes[0].bias = 0  # Output bias
        genome.nodes[1].bias = 0  # First hidden bias
        genome.nodes[2].bias = 0  # Second hidden bias
        genome.connections = dict()
        for i, o in [(-1, 1), (1, 2), (2, 0)]:
            genome.create_connection(config=config.genome_config, input_key=i, output_key=o, weight=1.0)
        if debug: print(genome)
        
        # Create a network
        game_config = GameConfig()
        net = make_net(genome=genome, config=config, game_config=game_config, bs=1)
        
        # Query the network; single input in range [-1, 1]
        inputs = [random() * 2 - 1 for _ in range(100)]
        hidden1_values = [relu_activation(torch.tensor(i, dtype=torch.float64)) for i in inputs]
        hidden2_values = [torch.tensor(0, dtype=torch.float64)] + \
                         [relu_activation(i) for i in hidden1_values[:-1]]
        output_values = [tanh_activation(h) for h in hidden2_values]
        
        # Query the network for the values
        for idx, inp in enumerate(inputs):
            [[r]] = query_net(net, [[inp]])
            self.assertAlmostEqual(r,
                                   float(output_values[idx]),
                                   delta=EPSILON)
    
    def test_2inp_1out(self, debug=False):
        """> Test single feedforward network with two inputs and one output.
        
        :note: Bias will be put to zero, and connection weights to 1.
        
        This test will check on the aggregation function of the output node.

        Network:
            I1 -          (-1) -
                |               |
                +- O  ==        +- (0)
                |               |
            I2 -          (-2) -
        """
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the genome and its corresponding config file
        genome, config = get_genome(1)
        
        # Manipulate the genome's biases and connection weights
        genome.nodes[0].bias = 0  # Output bias
        genome.connections = dict()
        for i, o in [(-1, 0), (-2, 0)]:
            genome.create_connection(config=config.genome_config, input_key=i, output_key=o, weight=1.0)
        if debug: print(genome)
        
        # Create a network
        game_config = GameConfig()
        net = make_net(genome=genome, config=config, game_config=game_config, bs=1)
        
        # Query the network; double inputs in range [-1, 1]
        inputs = [[random() * 2 - 1, random() * 2 - 1] for _ in range(100)]
        output_values = [tanh_activation(torch.tensor(i[0] + i[1], dtype=torch.float64)) for i in inputs]
        
        # Query the network for the values
        for idx, inp in enumerate(inputs):
            [[r]] = query_net(net, [[inp[0], inp[1]]])
            self.assertAlmostEqual(r,
                                   float(output_values[idx]),
                                   delta=EPSILON)
    
    def test_1inp_2hid_parallel_1out(self, debug=False):
        """> Test single feedforward network with one input, two hidden nodes in parallel, and one output.
        
        :note: Bias will be put to zero, and connection weights to 1.
        
        This test will check on the aggregation function of the output node, which should be doubled in value from its
        inputs.

        Network:
               +- H1 -+               +- (1) -+
               |      |               |       |
            I -+      +- O  ==  (-1) -+       +- (0)
               |      |               |       |
               +- H2 -+               +- (2) -+
        """
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the genome and its corresponding config file
        genome, config = get_genome(1)
        genome.nodes[1] = genome.create_node(config=config.genome_config, node_id=1)
        genome.nodes[2] = genome.create_node(config=config.genome_config, node_id=2)
        
        # Manipulate the genome's biases and connection weights
        genome.nodes[0].bias = 0  # Output bias
        genome.nodes[1].bias = 0  # Hidden bias
        genome.nodes[2].bias = 0  # Hidden bias
        genome.connections = dict()
        for i, o in [(-1, 1), (-1, 2), (1, 0), (2, 0)]:
            genome.create_connection(config=config.genome_config, input_key=i, output_key=o, weight=1.0)
        if debug: print(genome)
        
        # Create a network
        game_config = GameConfig()
        net = make_net(genome=genome, config=config, game_config=game_config, bs=1)
        
        # Query the network; only positive inputs (since relu simply forwards if positive)
        inputs = [random() for _ in range(100)]
        output_values = [tanh_activation(torch.tensor(2 * i, dtype=torch.float64)) for i in inputs]
        
        # Query the network for the values
        for idx, inp in enumerate(inputs):
            [[r]] = query_net(net, [[inp]])
            self.assertAlmostEqual(r,
                                   float(output_values[idx]),
                                   delta=EPSILON)


def main():
    # Test the feedforward network that contains only simple hidden nodes
    ff = TestFeedForward()
    ff.test_1inp_1out()
    ff.test_1inp_1hid_1out()
    ff.test_1inp_2hid_1out()
    ff.test_1inp_2hid_parallel_1out()
    ff.test_2inp_1out()


if __name__ == '__main__':
    unittest.main()
