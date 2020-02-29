"""
gru.py

Test if the GruNodeGene is implemented correctly
"""
import copy
import unittest

import torch

from configs.config import NeatConfig
from population.utils.config.default_config import Config
from population.utils.genome_util.genes import GruNodeGene
from population.utils.genome_util.genome import DefaultGenome
from population.utils.population_util.reproduction import DefaultReproduction
from population.utils.population_util.species import DefaultSpeciesSet
from population.utils.population_util.stagnation import DefaultStagnation


def get_config(num_inputs=4, num_hidden=1, num_outputs=1):
    """Create and return the config object."""
    cfg = NeatConfig()
    cfg.num_inputs = num_inputs
    cfg.num_hidden = num_hidden
    cfg.num_outputs = num_outputs
    cfg.initial_connection = "full_nodirect"  # input->hidden and hidden->output
    return Config(
            genome_type=DefaultGenome,
            reproduction_type=DefaultReproduction,
            species_set_type=DefaultSpeciesSet,
            stagnation_type=DefaultStagnation,
            config=cfg,
    )


def create_simple_genome(config):  # TODO: Delete!
    """
    A simple genome has one input, one output, and one hidden recurrent node (GRU). All weight parameters are forced to
    be 1, where all biases are forced to be 0.
    """
    g = DefaultGenome(key=1)
    
    # Init with randomized configuration
    g.configure_new(config.genome_config)
    
    # Node -1: input
    # --> No change needed
    # Node 0: output
    g.nodes[0].bias = 0
    # Node 1: GRU
    GRU_KEY = 1
    g.nodes[GRU_KEY] = g.create_gru_node(config.genome_config, 1)
    # g.nodes[GRU_KEY].bias_ih[:] = torch.FloatTensor([0, 0, 0])
    # g.nodes[GRU_KEY].bias_hh[:] = torch.FloatTensor([0, 0, 0])
    # g.nodes[GRU_KEY].weight_ih[:] = torch.FloatTensor([[0], [0], [0]])
    # g.nodes[GRU_KEY].weight_hh[:] = torch.FloatTensor([[0], [0], [0]])
    g.nodes[2].bias = 0
    
    # Connections
    for c in g.connections.values(): c.weight = 1
    return g


class TestGruNodeGene(unittest.TestCase):
    def test_input_keys(self):
        """Test if the input_keys list is expanded and contracted correctly."""
        gru = GruNodeGene(0)
        config = get_config()
        gru.init_attributes(config.genome_config)
        
        # Initial: both lists are empty
        assert len(gru.input_keys) == len(gru.full_input_keys) == 0
        
        # Add connection
        gru.add_input(config.genome_config, 1)
        assert len(gru.input_keys) == len(gru.full_input_keys) == 1
        
        # Remove non-existing connection --> changes nothing
        gru.remove_input(2)
        assert len(gru.input_keys) == len(gru.full_input_keys) == 1
        
        # Add connection with lower key
        gru.add_input(config.genome_config, -1)
        assert len(gru.input_keys) == len(gru.full_input_keys) == 2
        assert gru.input_keys == gru.full_input_keys == [-1, 1]
        
        # Add connection with key in the middle
        gru.add_input(config.genome_config, 0)
        assert len(gru.input_keys) == len(gru.full_input_keys) == 3
        assert gru.input_keys == gru.full_input_keys == [-1, 0, 1]
        
        # Add connection with key at the end
        gru.add_input(config.genome_config, 2)
        assert len(gru.input_keys) == len(gru.full_input_keys) == 4
        assert gru.input_keys == gru.full_input_keys == [-1, 0, 1, 2]
        
        # Remove existing connection
        gru.remove_input(1)
        assert len(gru.input_keys) == 3
        assert len(gru.full_input_keys) == 4
        assert gru.input_keys == [-1, 0, 2]
        assert gru.full_input_keys == [-1, 0, 1, 2]
        
        # Add previously added connection back
        gru.add_input(config.genome_config, 1)
        assert len(gru.input_keys) == len(gru.full_input_keys) == 4
        assert gru.input_keys == gru.full_input_keys == [-1, 0, 1, 2]
    
    def test_weight_ih(self):
        """Test if the weight_ih tensor is expanded and contracted correctly."""
        gru = GruNodeGene(0)
        config = get_config()
        gru.init_attributes(config.genome_config)
        gru.update_weight_ih()
        
        # Check if tensors initialized correctly
        assert type(gru.full_weight_ih) == torch.Tensor
        assert type(gru.weight_ih) == torch.Tensor
        assert gru.full_weight_ih.shape == (3, 0)
        assert gru.weight_ih.shape == (3, 0)
        
        # Add connection
        gru.add_input(config.genome_config, 1)
        gru.update_weight_ih()
        assert gru.full_weight_ih.shape == (3, 1)
        assert gru.weight_ih.shape == (3, 1)
        first_col = copy.deepcopy(gru.weight_ih[:, 0])
        assert all(gru.full_weight_ih[:, 0] == gru.weight_ih[:, 0])
        
        # Remove non-existing connection
        gru.remove_input(2)
        gru.update_weight_ih()
        assert gru.full_weight_ih.shape == (3, 1)
        assert gru.weight_ih.shape == (3, 1)
        assert all(gru.full_weight_ih[:, 0] == gru.weight_ih[:, 0])
        
        # Add second key
        gru.add_input(config.genome_config, 3)
        gru.update_weight_ih()
        assert gru.full_weight_ih.shape == (3, 2)
        assert gru.weight_ih.shape == (3, 2)
        assert all(gru.full_weight_ih[:, 0] == gru.weight_ih[:, 0]) and all(gru.weight_ih[:, 0] == first_col)
        third_col = copy.deepcopy(gru.weight_ih[:, 1])
        assert all(gru.full_weight_ih[:, 1] == gru.weight_ih[:, 1])
        
        # Add third key, positioned in between the two other keys
        gru.add_input(config.genome_config, 2)
        gru.update_weight_ih()
        assert gru.full_weight_ih.shape == (3, 3)
        assert gru.weight_ih.shape == (3, 3)
        assert all(gru.full_weight_ih[:, 0] == gru.weight_ih[:, 0]) and all(gru.weight_ih[:, 0] == first_col)
        second_col = copy.deepcopy(gru.weight_ih[:, 1])
        assert all(gru.full_weight_ih[:, 1] == gru.weight_ih[:, 1])
        assert all(gru.full_weight_ih[:, 2] == gru.weight_ih[:, 2]) and all(gru.weight_ih[:, 2] == third_col)
        
        # Remove the first connection
        gru.remove_input(1)
        gru.update_weight_ih()
        assert gru.full_weight_ih.shape == (3, 3)
        assert gru.weight_ih.shape == (3, 2)
        assert all(gru.full_weight_ih[:, 0] == first_col)
        assert all(gru.full_weight_ih[:, 1] == gru.weight_ih[:, 0]) and all(gru.weight_ih[:, 0] == second_col)
        assert all(gru.full_weight_ih[:, 2] == gru.weight_ih[:, 1]) and all(gru.weight_ih[:, 1] == third_col)
        
        # Add the first connection back
        gru.add_input(config.genome_config, 1)
        gru.update_weight_ih()
        assert gru.full_weight_ih.shape == (3, 3)
        assert gru.weight_ih.shape == (3, 3)
        assert all(gru.full_weight_ih[:, 0] == gru.weight_ih[:, 0]) and all(gru.weight_ih[:, 0] == first_col)
        assert all(gru.full_weight_ih[:, 1] == gru.weight_ih[:, 1]) and all(gru.weight_ih[:, 1] == second_col)
        assert all(gru.full_weight_ih[:, 2] == gru.weight_ih[:, 2]) and all(gru.weight_ih[:, 2] == third_col)
    
    def test_mutate(self):
        """Unused keys' values may never be mutated."""
        gru = GruNodeGene(0)
        config = get_config()
        gru.init_attributes(config.genome_config)
        gru.add_input(config.genome_config, 1)
        gru.add_input(config.genome_config, 2)
        gru.add_input(config.genome_config, 3)
        gru.add_input(config.genome_config, 4)
        gru.remove_input(2)
        gru.remove_input(4)
        gru.update_weight_ih()
        
        # Get current configuration
        first_col = copy.deepcopy(gru.full_weight_ih[:, 0])
        second_col = copy.deepcopy(gru.full_weight_ih[:, 1])
        third_col = copy.deepcopy(gru.full_weight_ih[:, 2])
        forth_col = copy.deepcopy(gru.full_weight_ih[:, 3])
        
        # Mutate a lot
        for _ in range(100): gru.mutate(config.genome_config)
        
        # Append second column back
        gru.add_input(config.genome_config, 2)
        gru.add_input(config.genome_config, 4)
        gru.update_weight_ih()
        
        # Check if second and forth columns haven't changed
        assert all(gru.weight_ih[:, 1] == second_col)
        assert all(gru.weight_ih[:, 3] == forth_col)
        
        # Check if other two columns have changed (almost impossible to obtain exactly the same after 100x mutation)
        assert not all(gru.weight_ih[:, 0] == first_col)
        assert not all(gru.weight_ih[:, 2] == third_col)


if __name__ == '__main__':
    unittest.main()

"""
if __name__ == '__main__':
    os.chdir("..")
    config = get_config()
    genome = create_simple_genome(config)
    print(genome, end="\n" * 3)
    # print(genome.input_keys)
    net = make_net(genome, config, bs=1, cold_start=True)
    
    # Query the network
    print("Querying the network:")
    inp = query_net(net, [[1, 1, 1, 1]])
    # inp = query_net(net, [[1, 1, 1, 1], [1, 1, 1, 1]])
    print(f" - iteration1: {inp}")
    inp = query_net(net, [[1, 1, 1, 1]])
    # inp = query_net(net, [[1, 1, 1, 1], [1, 1, 1, 1]])
    print(f" - iteration2: {inp}")
    inp = query_net(net, [[1, 1, 1, 1]])
    # inp = query_net(net, [[1, 1, 1, 1], [1, 1, 1, 1]])
    print(f" - iteration3: {inp}")
    # for _ in range(10):
    #     print(query_net(net, [[0]]))
"""
