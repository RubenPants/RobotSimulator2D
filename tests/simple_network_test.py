"""
network_test.py

Test the possible network configurations for the simple networks (i.e. feedforward network with simple hidden nodes).
"""
import os
import unittest
from random import random

import torch

from configs.config import NeatConfig
from population.population import query_net
from population.utils.config.default_config import Config
from population.utils.genome_util.genome import DefaultGenome
from population.utils.network_util.activations import relu_activation, tanh_activation
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
    cfg.initial_connection = D_FULL_NODIRECT
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
    def test_1inp_1out(self, debug=False):
        """
        Test the functionality of a simple network with only one input, one output, and no hidden nodes.
        Bias will be put to zero, and connection weights to 1.
        
        In this test, the value of the input will be mapped directly onto the output, where it will be squished by the
        output's squishing function (tanh).
        
        Network:
            I -- O  ==  (-1) -- (0)
        """
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the genome and its corresponding config file
        genome, config = get_genome(1, 0, 1)
        
        # Manipulate the genome's biases and connection weights
        genome.nodes[0].bias = 0  # Output-bias
        genome.connections[(-1, 0)].weight = 1  # Single connection from input to output
        if debug: print(genome)
        
        # Create a network
        net = make_net(genome, config, bs=1, cold_start=True)
        
        # Query the network; each input is directly mapped on the output (under tanh activation function)
        for _ in range(100):
            r = random() * 2 - 1
            o = query_net(net, [[r]])
            assert float(tanh_activation(torch.tensor(r, dtype=torch.float64))) == o
    
    def test_1inp_1hid_1out(self, debug=False):
        """
        Test the functionality of a simple network with only one input, one output, and one hidden node.
        Bias will be put to zero, and connection weights to 1.
        
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
        genome, config = get_genome(1, 1, 1)
        
        # Manipulate the genome's biases and connection weights
        genome.nodes[0].bias = 0  # Output-bias
        genome.nodes[1].bias = 0  # Hidden-bias
        genome.connections[(-1, 1)].weight = 1  # Single connection from input to hidden
        genome.connections[(1, 0)].weight = 1  # Single connection from hidden to output
        if debug: print(genome)
        
        # Create a network
        net = make_net(genome, config, bs=1, cold_start=True)
        
        # Query the network; single input in range [-1, 1]
        inputs = [random() * 2 - 1 for _ in range(100)]
        hidden_values = [relu_activation(torch.tensor(i, dtype=torch.float64)) for i in inputs]
        output_values = [tanh_activation(h) for h in hidden_values]
        
        # Query the network for the values
        for idx, inp in enumerate(inputs):
            [[r]] = query_net(net, [[inp]])
            assert r == float(output_values[idx])
    
    def test_1inp_2hid_1out(self, debug=False):
        """
        Test the functionality of a simple network with only one input, one output, and two hidden nodes.
        Bias will be put to zero, and connection weights to 1.
        
        For this test, in contrast to 'test_1inp_1hid_1out', the outputs aren't directly mapped from the inputs, but
        are delayed with one time-step (due to the double hidden nodes).

        Network:
            I -- H -- H -- O  ==  (-1) -- (1) -- (2) -- (0)
        """
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the genome and its corresponding config file
        genome, config = get_genome(1, 2, 1)
        
        # Manipulate the genome's biases and connection weights
        genome.nodes[0].bias = 0  # Output bias
        genome.nodes[1].bias = 0  # First hidden bias
        genome.nodes[2].bias = 0  # Second hidden bias
        del genome.connections[(-1, 2)]  # Unwanted connection
        del genome.connections[(1, 0)]  # Unwanted connection
        genome.create_connection(config=config.genome_config, input_key=1, output_key=2)
        genome.connections[(-1, 1)].weight = 1  # Single connection from input to first hidden
        genome.connections[(1, 2)].weight = 1  # Single connection from first hidden to second hidden
        genome.connections[(2, 0)].weight = 1  # Single connection from second hidden to output
        if debug: print(genome)
        
        # Create a network
        net = make_net(genome, config, bs=1, cold_start=True)
        
        # Query the network; single input in range [-1, 1]
        inputs = [random() * 2 - 1 for _ in range(100)]
        hidden1_values = [relu_activation(torch.tensor(i, dtype=torch.float64)) for i in inputs]
        hidden2_values = [torch.tensor(0, dtype=torch.float64)] + \
                         [relu_activation(i) for i in hidden1_values[:-1]]
        output_values = [tanh_activation(h) for h in hidden2_values]
        
        # Query the network for the values
        for idx, inp in enumerate(inputs):
            [[r]] = query_net(net, [[inp]])
            assert r == float(output_values[idx])
    
    def test_2inp_1out(self, debug=False):
        """
        Test the functionality of a simple network with two inputs and one output.
        Bias will be put to zero, and connection weights to 1.
        
        This test will check on the aggregation function of the output node.

        Network:
            I1 -         (-1) -
                |              |
                +- O  ==       +- (0)
                |              |
            I2 -         (-2) -
        """
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the genome and its corresponding config file
        genome, config = get_genome(2, 0, 1)
        
        # Manipulate the genome's biases and connection weights
        genome.nodes[0].bias = 0  # Output bias
        genome.connections[(-1, 0)].weight = 1  # Single connection from input to first hidden
        genome.connections[(-2, 0)].weight = 1  # Single connection from input to first hidden
        if debug: print(genome)
        
        # Create a network
        net = make_net(genome, config, bs=1, cold_start=True)
        
        # Query the network; double inputs in range [-1, 1]
        inputs = [[random() * 2 - 1, random() * 2 - 1] for _ in range(100)]
        output_values = [tanh_activation(torch.tensor(i[0] + i[1], dtype=torch.float64)) for i in inputs]
        
        # Query the network for the values
        for idx, inp in enumerate(inputs):
            [[r]] = query_net(net, [[inp[0], inp[1]]])
            assert r == float(output_values[idx])
    
    def test_1inp_2hid_parallel_1out(self, debug=False):
        """
        Test the functionality of a simple network with one input, two hidden in parallel, and one output.
        Bias will be put to zero, and connection weights to 1.
        
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
        genome, config = get_genome(1, 2, 1)
        
        # Manipulate the genome's biases and connection weights
        genome.nodes[0].bias = 0  # Output bias
        genome.nodes[1].bias = 0  # Output bias
        genome.nodes[2].bias = 0  # Output bias
        genome.connections[(-1, 1)].weight = 1  # Single connection from input to first hidden
        genome.connections[(-1, 2)].weight = 1  # Single connection from input to second hidden
        genome.connections[(1, 0)].weight = 1  # Single connection from first hidden to output
        genome.connections[(2, 0)].weight = 1  # Single connection from second hidden to output
        if debug: print(genome)
        
        # Create a network
        net = make_net(genome, config, bs=1, cold_start=True)
        
        # Query the network; only positive inputs (since relu simply forwards if positive)
        inputs = [random() for _ in range(100)]
        output_values = [tanh_activation(torch.tensor(2 * i, dtype=torch.float64)) for i in inputs]
        
        # Query the network for the values
        for idx, inp in enumerate(inputs):
            [[r]] = query_net(net, [[inp]])
            assert r == float(output_values[idx])


def main():
    success, fail = 0, 0
    
    # Test the feedforward network that contains only simple hidden nodes
    ff = TestFeedForward()
    try:
        ff.test_1inp_1out()
        success += 1
    except AssertionError:
        fail += 1
    try:
        ff.test_1inp_1hid_1out()
        success += 1
    except AssertionError:
        fail += 1
    try:
        ff.test_1inp_2hid_1out()
        success += 1
    except AssertionError:
        fail += 1
    try:
        ff.test_1inp_2hid_parallel_1out()
        success += 1
    except AssertionError:
        fail += 1
    try:
        ff.test_2inp_1out()
        success += 1
    except AssertionError:
        fail += 1
        
    return success, fail


if __name__ == '__main__':
    unittest.main()
