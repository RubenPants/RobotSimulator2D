"""
gru_network_test.py

Test the possible network configurations for networks that contain a GRU hidden node.
"""
import os
import unittest
from random import random

import torch

from config import Config
from population.population import query_net
from population.utils.genome_util.genome import DefaultGenome
from population.utils.network_util.activations import tanh_activation
from population.utils.network_util.feed_forward_net import make_net
from utils.dictionary import *

# Precision
EPSILON = 1e-5


def get_genome(outputs):
    """Create a simple feedforward neuron."""  # Get the configuration
    cfg = Config()
    cfg.genome.num_outputs = outputs
    cfg.genome.initial_connection = D_FULL_NODIRECT  # input -> hidden -> output
    cfg.genome.gru_enabled = True  # Only simple hidden nodes allowed
    cfg.genome.gru_node_prob = 1.0  # Force that all hidden nodes will be GRUs
    cfg.update()
    
    # Create the genome
    g = DefaultGenome(key=0, num_outputs=cfg.genome.num_outputs)
    g.configure_new(cfg.genome)
    return g, cfg


class TestGruFeedForward(unittest.TestCase):
    def test_1inp_1hid_1out(self, debug=False):
        """> Test single feedforward network with one input, one hidden GRU node, and one output.

        :note: Bias will be put to zero, and connection weights to 1. All the weights of the GRU node are set to 1.

        In this test, the value of the input progresses in three steps to reach the output:
          1) input-to-hidden: The value of the input node squished by the hidden node's relu function
          2) hidden: Update the GRU-nodes based on current input and (GRU) hidden state
          3) hidden-to-output: The value of the hidden node squished by the output node's tanh function
        This flow is executed in one go, since each iteration, all hidden nodes are updated before taking the output
        nodes into account.

        Network:
            I -- GRU -- O  ==  (-1) -- (1) -- (0)
        """
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the genome and its corresponding config file
        genome, config = get_genome(1)
        
        # Add the hidden GRU node
        genome.nodes[1] = genome.create_gru_node(config.genome, node_id=1)
        
        # Manipulate the genome's biases and connection weights
        genome.nodes[0].bias = 0  # Output-bias
        genome.connections = dict()
        for i, o in [(-1, 1), (1, 0)]:
            genome.create_connection(config=config.genome, input_key=i, output_key=o, weight=1.0)
        genome.update_gru_nodes(config=config.genome)
        genome.nodes[1].gru_bias_ih[:] = torch.tensor([1, 1, 1], dtype=torch.float64)
        genome.nodes[1].gru_bias_hh[:] = torch.tensor([1, 1, 1], dtype=torch.float64)
        genome.nodes[1].gru_full_weight_ih[:] = torch.tensor([[1], [1], [1]], dtype=torch.float64)
        genome.nodes[1].gru_weight_hh[:] = torch.tensor([[1], [1], [1]], dtype=torch.float64)
        if debug: print(genome)
        # Create a network
        net = make_net(genome=genome, genome_config=config.genome, game_config=config, bs=1)
        
        # Query the network; single input in range [-1, 1]
        inputs = [random() * 2 - 1 for _ in range(100)]
        gru = torch.nn.GRUCell(1, 1)
        gru.bias_ih[:] = genome.nodes[1].gru_bias_ih[:]
        gru.bias_hh[:] = genome.nodes[1].gru_bias_hh[:]
        gru.weight_ih[:] = genome.nodes[1].gru_weight_ih[:]
        gru.weight_hh[:] = genome.nodes[1].gru_weight_hh[:]
        hidden_values = []
        hx = None
        for inp in inputs:
            hx = gru(torch.FloatTensor([[inp]]), hx)
            hidden_values.append(hx)
        output_values = [tanh_activation(h) for h in hidden_values]
        
        # Query the network for the values
        for idx, inp in enumerate(inputs):
            [[r]] = query_net(net, [[inp]])
            self.assertAlmostEqual(r, float(output_values[idx]), delta=EPSILON)


def main():
    # Test the feedforward network that contains only simple hidden nodes
    gff = TestGruFeedForward()
    gff.test_1inp_1hid_1out()


if __name__ == '__main__':
    unittest.main()
