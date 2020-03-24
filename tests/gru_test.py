"""
gru.py

Test if the GruNodeGene is implemented correctly
"""
import copy
import os
import unittest

import torch

from config import Config
from population.utils.genome_util.genes import GruNodeGene


def get_config(num_outputs=1):
    """Create and return the config object."""
    cfg = Config()
    cfg.genome.num_outputs = num_outputs
    cfg.genome.initial_connection = "full_nodirect"  # input->hidden and hidden->output
    cfg.update()
    return cfg


class TestGruNodeGene(unittest.TestCase):
    def test_input_keys(self):
        """> Test if the input_keys list is expanded and contracted correctly."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        gru = GruNodeGene(0)
        config = get_config()
        gru.init_attributes(config.genome)
        
        # Initial: both lists are empty
        self.assertTrue(len(gru.input_keys) == len(gru.full_input_keys) == 0)
        
        # Add connection
        gru.add_input(config.genome, 1)
        self.assertTrue(len(gru.input_keys) == len(gru.full_input_keys) == 1)
        
        # Remove non-existing connection --> changes nothing
        gru.remove_input(2)
        self.assertTrue(len(gru.input_keys) == len(gru.full_input_keys) == 1)
        
        # Add connection with lower key
        gru.add_input(config.genome, -1)
        self.assertTrue(len(gru.input_keys) == len(gru.full_input_keys) == 2)
        self.assertTrue(gru.input_keys == gru.full_input_keys == [-1, 1])
        
        # Add connection with key in the middle
        gru.add_input(config.genome, 0)
        self.assertTrue(len(gru.input_keys) == len(gru.full_input_keys) == 3)
        self.assertTrue(gru.input_keys == gru.full_input_keys == [-1, 0, 1])
        
        # Add connection with key at the end
        gru.add_input(config.genome, 2)
        self.assertTrue(len(gru.input_keys) == len(gru.full_input_keys) == 4)
        self.assertTrue(gru.input_keys == gru.full_input_keys == [-1, 0, 1, 2])
        
        # Remove existing connection
        gru.remove_input(1)
        self.assertTrue(len(gru.input_keys) == 3)
        self.assertTrue(len(gru.full_input_keys) == 4)
        self.assertTrue(gru.input_keys == [-1, 0, 2])
        self.assertTrue(gru.full_input_keys == [-1, 0, 1, 2])
        
        # Add previously added connection back
        gru.add_input(config.genome, 1)
        self.assertTrue(len(gru.input_keys) == len(gru.full_input_keys) == 4)
        self.assertTrue(gru.input_keys == gru.full_input_keys == [-1, 0, 1, 2])
    
    def test_weight_ih(self):
        """> Test if the weight_ih tensor is expanded and contracted correctly."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        gru = GruNodeGene(0)
        config = get_config()
        gru.init_attributes(config.genome)
        gru.update_weight_ih()
        
        # Check if tensors initialized correctly
        self.assertTrue(type(gru.gru_full_weight_ih) == torch.Tensor)
        self.assertTrue(type(gru.gru_weight_ih) == torch.Tensor)
        self.assertTrue(gru.gru_full_weight_ih.shape == (3, 0))
        self.assertTrue(gru.gru_weight_ih.shape == (3, 0))
        
        # Add connection
        gru.add_input(config.genome, 1)
        gru.update_weight_ih()
        self.assertTrue(gru.gru_full_weight_ih.shape == (3, 1))
        self.assertTrue(gru.gru_weight_ih.shape == (3, 1))
        first_col = copy.deepcopy(gru.gru_weight_ih[:, 0])
        self.assertTrue(all(gru.gru_full_weight_ih[:, 0] == gru.gru_weight_ih[:, 0]))
        
        # Remove non-existing connection
        gru.remove_input(2)
        gru.update_weight_ih()
        self.assertTrue(gru.gru_full_weight_ih.shape == (3, 1))
        self.assertTrue(gru.gru_weight_ih.shape == (3, 1))
        self.assertTrue(all(gru.gru_full_weight_ih[:, 0] == gru.gru_weight_ih[:, 0]))
        
        # Add second key
        gru.add_input(config.genome, 3)
        gru.update_weight_ih()
        self.assertTrue(gru.gru_full_weight_ih.shape == (3, 2))
        self.assertTrue(gru.gru_weight_ih.shape == (3, 2))
        self.assertTrue(all(gru.gru_full_weight_ih[:, 0] == gru.gru_weight_ih[:, 0]))
        self.assertTrue(all(gru.gru_weight_ih[:, 0] == first_col))
        third_col = copy.deepcopy(gru.gru_weight_ih[:, 1])
        self.assertTrue(all(gru.gru_full_weight_ih[:, 1] == gru.gru_weight_ih[:, 1]))
        
        # Add third key, positioned in between the two other keys
        gru.add_input(config.genome, 2)
        gru.update_weight_ih()
        self.assertTrue(gru.gru_full_weight_ih.shape == (3, 3))
        self.assertTrue(gru.gru_weight_ih.shape == (3, 3))
        self.assertTrue(all(gru.gru_full_weight_ih[:, 0] == gru.gru_weight_ih[:, 0]))
        self.assertTrue(all(gru.gru_weight_ih[:, 0] == first_col))
        second_col = copy.deepcopy(gru.gru_weight_ih[:, 1])
        self.assertTrue(all(gru.gru_full_weight_ih[:, 1] == gru.gru_weight_ih[:, 1]))
        self.assertTrue(all(gru.gru_full_weight_ih[:, 2] == gru.gru_weight_ih[:, 2]))
        self.assertTrue(all(gru.gru_weight_ih[:, 2] == third_col))
        
        # Remove the first connection
        gru.remove_input(1)
        gru.update_weight_ih()
        self.assertTrue(gru.gru_full_weight_ih.shape == (3, 3))
        self.assertTrue(gru.gru_weight_ih.shape == (3, 2))
        self.assertTrue(all(gru.gru_full_weight_ih[:, 0] == first_col))
        self.assertTrue(all(gru.gru_full_weight_ih[:, 1] == gru.gru_weight_ih[:, 0]))
        self.assertTrue(all(gru.gru_weight_ih[:, 0] == second_col))
        self.assertTrue(all(gru.gru_full_weight_ih[:, 2] == gru.gru_weight_ih[:, 1]))
        self.assertTrue(all(gru.gru_weight_ih[:, 1] == third_col))
        
        # Add the first connection back
        gru.add_input(config.genome, 1)
        gru.update_weight_ih()
        self.assertTrue(gru.gru_full_weight_ih.shape == (3, 3))
        self.assertTrue(gru.gru_weight_ih.shape == (3, 3))
        self.assertTrue(all(gru.gru_full_weight_ih[:, 0] == gru.gru_weight_ih[:, 0]))
        self.assertTrue(all(gru.gru_weight_ih[:, 0] == first_col))
        self.assertTrue(all(gru.gru_full_weight_ih[:, 1] == gru.gru_weight_ih[:, 1]))
        self.assertTrue(all(gru.gru_weight_ih[:, 1] == second_col))
        self.assertTrue(all(gru.gru_full_weight_ih[:, 2] == gru.gru_weight_ih[:, 2]))
        self.assertTrue(all(gru.gru_weight_ih[:, 2] == third_col))
    
    def test_mutate(self):
        """> Unused keys' values may never be mutated."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        gru = GruNodeGene(0)
        config = get_config()
        gru.init_attributes(config.genome)
        gru.add_input(config.genome, 1)
        gru.add_input(config.genome, 2)
        gru.add_input(config.genome, 3)
        gru.add_input(config.genome, 4)
        gru.remove_input(2)
        gru.remove_input(4)
        gru.update_weight_ih()
        
        # Get current configuration
        first_col = copy.deepcopy(gru.gru_full_weight_ih[:, 0])
        second_col = copy.deepcopy(gru.gru_full_weight_ih[:, 1])
        third_col = copy.deepcopy(gru.gru_full_weight_ih[:, 2])
        forth_col = copy.deepcopy(gru.gru_full_weight_ih[:, 3])
        
        # Mutate a lot
        for _ in range(100): gru.mutate(config.genome)
        
        # Append second column back
        gru.add_input(config.genome, 2)
        gru.add_input(config.genome, 4)
        gru.update_weight_ih()
        
        # Check if second and forth columns haven't changed
        self.assertTrue(all(gru.gru_weight_ih[:, 1] == second_col))
        self.assertTrue(all(gru.gru_weight_ih[:, 3] == forth_col))
        
        # Check if other two columns have changed (almost impossible to obtain exactly the same after 100x mutation)
        self.assertTrue(not all(gru.gru_weight_ih[:, 0] == first_col))
        self.assertTrue(not all(gru.gru_weight_ih[:, 2] == third_col))


def main():
    # Test basic GruNodeGene test cases
    gng = TestGruNodeGene()
    gng.test_input_keys()
    gng.test_weight_ih()
    gng.test_mutate()


if __name__ == '__main__':
    unittest.main()
