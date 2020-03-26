"""
genome_mutation_test.py

Test gene-specific operations.
"""
import os
import unittest

import numpy as np

from config import Config
from population.utils.genome_util.genes import ConnectionGene, GruNodeGene, OutputNodeGene, SimpleNodeGene
from utils.dictionary import *


def get_connection_gene(key, config):
    return ConnectionGene(key, config)


def get_gru_node_gene(key, config):
    return GruNodeGene(key, config, input_keys=[-1], input_keys_full=[-1, -2])


def get_output_node_gene(key, config):
    return OutputNodeGene(key, config)


def get_simple_node_gene(key, config):
    return SimpleNodeGene(key, config)


class SimpleNode(unittest.TestCase):
    """Test the SimpleNodeGene's mutation operations."""
    
    def test_activation(self):
        """> Test if activation changes during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, activation is always one of the options
        cfg = Config().genome
        cfg.activation_default = "a"
        cfg.activation_mutate_rate = 1.0
        OPTIONS = {"a": 1, "b": 2, "c": 3}
        cfg.activation_options = OPTIONS
        gene = get_simple_node_gene(0, cfg)
        changed = False
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.activation in OPTIONS.keys())
            if gene.activation != "a": changed = True
        self.assertTrue(changed)  # Almost impossible that this failed
        
        # Set mutation rate to 0, activation should not mutate
        cfg.activation_default = "a"
        cfg.activation_mutate_rate = 0.0
        cfg.activation_options = OPTIONS
        gene = get_simple_node_gene(0, cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.activation == 'a')
    
    def test_aggregation(self):
        """> Test if aggregation changes during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, aggregation is always one of the options
        cfg = Config().genome
        cfg.aggregation_default = "a"
        cfg.aggregation_mutate_rate = 1.0
        OPTIONS = {"a": 1, "b": 2, "c": 3}
        cfg.aggregation_options = OPTIONS
        gene = get_simple_node_gene(0, cfg)
        changed = False
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.aggregation in OPTIONS.keys())
            if gene.aggregation != "a": changed = True
        self.assertTrue(changed)  # Almost impossible that this failed
        
        # Set mutation rate to 0, aggregation should not mutate
        cfg.aggregation_default = "a"
        cfg.aggregation_mutate_rate = 0.0
        cfg.aggregation_options = OPTIONS
        gene = get_simple_node_gene(0, cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.aggregation == 'a')
    
    def test_bias(self):
        """> Test if the bias remains inside its boundaries during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, bias remains inside its boundary
        cfg = Config().genome
        cfg.bias_mutate_rate = 0.5
        cfg.bias_replace_rate = 0.5
        cfg.bias_min_value = -0.1
        cfg.bias_max_value = 0.1
        gene = get_simple_node_gene(0, cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(-0.1 <= gene.bias <= 0.1)
        
        # Set mutation rate to 0, no change should happen
        cfg.bias_mutate_rate = 0
        cfg.bias_replace_rate = 0
        gene = get_simple_node_gene(0, cfg)
        init_bias = gene.bias
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.bias == init_bias)
        
        # Set mutation power to 0, no change should happen
        cfg.bias_mutate_rate = 1
        cfg.bias_replace_rate = 0
        cfg.bias_mutate_power = 0
        gene = get_simple_node_gene(0, cfg)
        init_bias = gene.bias
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.bias == init_bias)


class OutputNode(unittest.TestCase):
    """Test the OutputNodeGene's mutation operations."""
    
    def test_activation(self):
        """> Test if activation remains tanh after mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, activation is always one of the options
        cfg = Config().genome
        cfg.activation_default = "a"
        cfg.activation_mutate_rate = 1.0
        OPTIONS = {"a": 1, "b": 2, "c": 3}
        cfg.activation_options = OPTIONS
        gene = get_output_node_gene(0, cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(gene.activation, D_TANH)
    
    def test_aggregation(self):
        """> Test if aggregation changes during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, aggregation is always one of the options
        cfg = Config().genome
        cfg.aggregation_default = "a"
        cfg.aggregation_mutate_rate = 1.0
        OPTIONS = {"a": 1, "b": 2, "c": 3}
        cfg.aggregation_options = OPTIONS
        gene = get_output_node_gene(0, cfg)
        changed = False
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.aggregation in OPTIONS.keys())
            if gene.aggregation != "a": changed = True
        self.assertTrue(changed)  # Almost impossible that this failed
        
        # Set mutation rate to 0, aggregation should not mutate
        cfg.aggregation_default = "a"
        cfg.aggregation_mutate_rate = 0.0
        cfg.aggregation_options = OPTIONS
        gene = get_output_node_gene(0, cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.aggregation == 'a')
    
    def test_bias(self):
        """> Test if the bias remains inside its boundaries during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, bias remains inside its boundary
        cfg = Config().genome
        cfg.bias_mutate_rate = 0.5
        cfg.bias_replace_rate = 0.5
        cfg.bias_min_value = -0.1
        cfg.bias_max_value = 0.1
        gene = get_output_node_gene(0, cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(-0.1 <= gene.bias <= 0.1)
        
        # Set mutation rate to 0, no change should happen
        cfg.bias_mutate_rate = 0
        cfg.bias_replace_rate = 0
        gene = get_output_node_gene(0, cfg)
        init_bias = gene.bias
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.bias == init_bias)
        
        # Set mutation power to 0, no change should happen
        cfg.bias_mutate_rate = 1
        cfg.bias_replace_rate = 0
        cfg.bias_mutate_power = 0
        gene = get_output_node_gene(0, cfg)
        init_bias = gene.bias
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.bias == init_bias)


class GruNode(unittest.TestCase):
    """Test the GruNodeGene's mutation operations."""
    
    def test_activation(self):
        """> Test if activation changes during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, activation is always one of the options
        cfg = Config().genome
        cfg.activation_default = "a"
        cfg.activation_mutate_rate = 1.0
        OPTIONS = {"a": 1, "b": 2, "c": 3}
        cfg.activation_options = OPTIONS
        gene = get_gru_node_gene(0, cfg)
        changed = False
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.activation in OPTIONS.keys())
            if gene.activation != "a": changed = True
        self.assertTrue(changed)  # Almost impossible that this failed
        
        # Set mutation rate to 0, activation should not mutate
        cfg.activation_default = "a"
        cfg.activation_mutate_rate = 0.0
        cfg.activation_options = OPTIONS
        gene = get_gru_node_gene(0, cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.activation == 'a')
    
    def test_bias(self):
        """> Test if bias is left unchanged during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, bias remains equal to zero
        cfg = Config().genome
        cfg.bias_mutate_rate = 0.5
        cfg.bias_replace_rate = 0.5
        gene = get_gru_node_gene(0, cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(gene.bias, 0)
    
    def test_bias_hh(self):
        """> Test if bias_hh behaves as expected"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, bias_hh's values remain inside the set boundary
        cfg = Config().genome
        cfg.gru_mutate_rate = 0.5
        cfg.gru_replace_rate = 0.5
        cfg.gru_min_value = -0.1
        cfg.gru_max_value = 0.1
        gene = get_gru_node_gene(0, cfg)
        changed = False
        init_bias_hh = gene.bias_hh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            for value in gene.bias_hh:
                self.assertTrue(-0.1 <= value <= 0.1)
            if np.linalg.norm(gene.bias_hh - init_bias_hh) > 0: changed = True
        self.assertTrue(changed)
        
        # Set mutation rate to 0, no change should happen
        cfg.gru_mutate_rate = 0
        cfg.gru_replace_rate = 0
        gene = get_gru_node_gene(0, cfg)
        init_bias_hh = gene.bias_hh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.bias_hh - init_bias_hh), 0)
        
        # Set mutation power to 0, no change should happen
        cfg.gru_mutate_rate = 1
        cfg.gru_replace_rate = 0
        cfg.gru_mutate_power = 0
        gene = get_gru_node_gene(0, cfg)
        init_bias_hh = gene.bias_hh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.bias_hh - init_bias_hh), 0)
    
    def test_bias_ih(self):
        """> Test if bias_ih behaves as expected"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, bias_ih's values remain inside the set boundary
        cfg = Config().genome
        cfg.gru_mutate_rate = 0.5
        cfg.gru_replace_rate = 0.5
        cfg.gru_min_value = -0.1
        cfg.gru_max_value = 0.1
        gene = get_gru_node_gene(0, cfg)
        changed = False
        init_bias_ih = gene.bias_ih.copy()
        for _ in range(100):
            gene.mutate(cfg)
            for value in gene.bias_ih:
                self.assertTrue(-0.1 <= value <= 0.1)
            if np.linalg.norm(gene.bias_ih - init_bias_ih) > 0: changed = True
        self.assertTrue(changed)
        
        # Set mutation rate to 0, no change should happen
        cfg.gru_mutate_rate = 0
        cfg.gru_replace_rate = 0
        gene = get_gru_node_gene(0, cfg)
        init_bias_ih = gene.bias_ih.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.bias_ih - init_bias_ih), 0)
        
        # Set mutation power to 0, no change should happen
        cfg.gru_mutate_rate = 1
        cfg.gru_replace_rate = 0
        cfg.gru_mutate_power = 0
        gene = get_gru_node_gene(0, cfg)
        init_bias_ih = gene.bias_ih.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.bias_ih - init_bias_ih), 0)
    
    def test_weight_hh(self):
        """> Test if weight_hh behaves as expected"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, weight_hh's values remain inside the set boundary
        cfg = Config().genome
        cfg.gru_mutate_rate = 0.5
        cfg.gru_replace_rate = 0.5
        cfg.gru_min_value = -0.1
        cfg.gru_max_value = 0.1
        gene = get_gru_node_gene(0, cfg)
        changed = False
        init_weight_hh = gene.weight_hh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            for value in gene.weight_hh:
                self.assertTrue(-0.1 <= value <= 0.1)
            if np.linalg.norm(gene.weight_hh - init_weight_hh) > 0: changed = True
        self.assertTrue(changed)
        
        # Set mutation rate to 0, no change should happen
        cfg.gru_mutate_rate = 0
        cfg.gru_replace_rate = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_hh = gene.weight_hh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.weight_hh - init_weight_hh), 0)
        
        # Set mutation power to 0, no change should happen
        cfg.gru_mutate_rate = 1
        cfg.gru_replace_rate = 0
        cfg.gru_mutate_power = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_hh = gene.weight_hh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.weight_hh - init_weight_hh), 0)
    
    def test_weight_ih(self):
        """> Test if weight_ih behaves as expected"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Test on shape of weight_ih
        cfg = Config().genome
        gene = get_gru_node_gene(0, cfg)
        self.assertEqual(gene.weight_ih.shape, (3, 1))
        
        # After mutations, weight_ih's values remain inside the set boundary
        cfg.gru_mutate_rate = 0.5
        cfg.gru_replace_rate = 0.5
        cfg.gru_min_value = -0.1
        cfg.gru_max_value = 0.1
        gene = get_gru_node_gene(0, cfg)
        changed = False
        init_weight_ih = gene.weight_ih.copy()
        for _ in range(100):
            gene.mutate(cfg)
            gene.update_weight_ih()
            for value in gene.weight_ih:
                for v in value:
                    self.assertTrue(-0.1 <= value <= 0.1)
            if np.linalg.norm(gene.weight_ih - init_weight_ih) > 0: changed = True
        self.assertTrue(changed)
        
        # Set mutation rate to 0, no change should happen
        cfg.gru_mutate_rate = 0
        cfg.gru_replace_rate = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_ih = gene.weight_ih.copy()
        for _ in range(100):
            gene.mutate(cfg)
            gene.update_weight_ih()
            self.assertEqual(np.linalg.norm(gene.weight_ih - init_weight_ih), 0)
        
        # Set mutation power to 0, no change should happen
        cfg.gru_mutate_rate = 1
        cfg.gru_replace_rate = 0
        cfg.gru_mutate_power = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_ih = gene.weight_ih.copy()
        for _ in range(100):
            gene.mutate(cfg)
            gene.update_weight_ih()
            self.assertEqual(np.linalg.norm(gene.weight_ih - init_weight_ih), 0)
    
    def test_weight_ih_full(self):
        """> Test if weight_ih_full behaves as expected"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Test on shape of weight_ih_full
        cfg = Config().genome
        gene = get_gru_node_gene(0, cfg)
        self.assertEqual(gene.weight_ih_full.shape, (3, 2))
        
        # After mutations, weight_ih_full's values remain inside the set boundary
        cfg.gru_mutate_rate = 0.5
        cfg.gru_replace_rate = 0.5
        cfg.gru_min_value = -0.1
        cfg.gru_max_value = 0.1
        gene = get_gru_node_gene(0, cfg)
        changed = False
        init_weight_ih_full = gene.weight_ih_full.copy()
        for _ in range(100):
            gene.mutate(cfg)  # No update_weight_ih must be called
            for value in gene.weight_ih_full:
                for v in value:
                    self.assertTrue(-0.1 <= v <= 0.1)
            if np.linalg.norm(gene.weight_ih_full - init_weight_ih_full) > 0: changed = True
        self.assertTrue(changed)
        
        # Set mutation rate to 0, no change should happen
        cfg.gru_mutate_rate = 0
        cfg.gru_replace_rate = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_ih_full = gene.weight_ih_full.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.weight_ih_full - init_weight_ih_full), 0)
        
        # Set mutation power to 0, no change should happen
        cfg.gru_mutate_rate = 1
        cfg.gru_replace_rate = 0
        cfg.gru_mutate_power = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_ih_full = gene.weight_ih_full.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.weight_ih_full - init_weight_ih_full), 0)


class Connection(unittest.TestCase):
    """Test the ConnectionGene's mutation operations."""
    
    def test_enabled(self):
        """> Test if enabled changes during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Test the enabled defaults
        cfg = Config().genome
        cfg.enabled_default = False
        gene = get_connection_gene((-1, 0), cfg)
        self.assertFalse(gene.enabled)
        cfg.enabled_default = True
        gene = get_connection_gene((-1, 0), cfg)
        self.assertTrue(gene.enabled)
        
        # Enabled state should change during mutation
        cfg.enabled_mutate_rate = 1
        changed = False
        gene = get_connection_gene((-1, 0), cfg)
        init_enabled = gene.enabled
        for _ in range(100):
            gene.mutate(cfg)
            if gene.enabled != init_enabled:
                changed = True
                break
        self.assertTrue(changed)
    
    def test_weight(self):
        """> Test if the weight remains inside its boundaries during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, weight remains inside its boundary
        cfg = Config().genome
        cfg.weight_mutate_rate = 0.5
        cfg.weight_replace_rate = 0.5
        cfg.weight_min_value = -0.1
        cfg.weight_max_value = 0.1
        gene = get_connection_gene((-1, 0), cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(-0.1 <= gene.weight <= 0.1)
        
        # Set mutation rate to 0, no change should happen
        cfg.weight_mutate_rate = 0
        cfg.weight_replace_rate = 0
        gene = get_connection_gene((-1, 0), cfg)
        init_weight = gene.weight
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.weight == init_weight)
        
        # Set mutation power to 0, no change should happen
        cfg.weight_mutate_rate = 1
        cfg.weight_replace_rate = 0
        cfg.weight_mutate_power = 0
        gene = get_connection_gene((-1, 0), cfg)
        init_weight = gene.weight
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.weight == init_weight)


def main():
    # Test the SimpleNodeGene
    sn = SimpleNode()
    sn.test_activation()
    sn.test_aggregation()
    sn.test_bias()
    
    # Test the OutputNodeGene
    on = OutputNode()
    on.test_activation()
    on.test_aggregation()
    on.test_bias()
    
    # Test the GruNodeGene
    gn = GruNode()
    gn.test_activation()
    gn.test_bias()
    gn.test_bias_hh()
    gn.test_bias_ih()
    gn.test_weight_hh()
    gn.test_weight_ih()
    gn.test_weight_ih_full()
    
    # Test the ConnectionGene
    c = Connection()
    c.test_enabled()
    c.test_weight()


if __name__ == '__main__':
    unittest.main()
