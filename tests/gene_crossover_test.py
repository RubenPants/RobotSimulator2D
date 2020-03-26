"""
genome_crossover_test.py

Test gene-specific operations.
"""
import os
import unittest
from copy import deepcopy

import numpy as np

from config import Config
from population.utils.genome_util.genes import ConnectionGene, GruNodeGene, OutputNodeGene, SimpleNodeGene
from utils.dictionary import D_TANH


def get_connection_genes(key, config):
    """Create two connection genes, first enabled with weight set to zero, and the other disabled with a weight of 1."""
    gene1 = ConnectionGene(key, config)
    gene1.enabled = True
    gene1.weight = 0
    gene2 = ConnectionGene(key, config)
    gene2.enabled = False
    gene2.weight = 1
    return gene1, gene2


def get_gru_node_gene(key, config):
    """
    Create two GRU genes, one initialized with all zeros, where the other is initialized with all ones. For the
    weight_ih layer, they have both two input_keys_full, but only one shared input_keys.
    """
    gene1 = GruNodeGene(key, config, input_keys=[-1], input_keys_full=[-1, -2])
    gene1.activation = 'a'
    gene1.bias_hh = np.zeros(gene1.bias_hh.shape)
    gene1.bias_ih = np.zeros(gene1.bias_ih.shape)
    gene1.weight_hh = np.zeros(gene1.weight_hh.shape)
    gene1.weight_ih_full = np.zeros(gene1.weight_ih_full.shape)
    gene1.update_weight_ih()
    gene2 = GruNodeGene(key, config, input_keys=[-1], input_keys_full=[-1, -3])
    gene2.activation = 'b'
    gene2.bias_hh = np.ones(gene2.bias_hh.shape)
    gene2.bias_ih = np.ones(gene2.bias_ih.shape)
    gene2.weight_hh = np.ones(gene2.weight_hh.shape)
    gene2.weight_ih_full = np.ones(gene2.weight_ih_full.shape)
    gene2.update_weight_ih()
    return gene1, gene2


def get_output_node_gene(key, config):
    """
    Create two output genes, one enabled with ('a', 0) and the other with ('b', 1) for (aggregation, bias) respectively.
    """
    gene1 = OutputNodeGene(key, config)
    gene1.aggregation = 'a'
    gene1.bias = 0
    gene2 = OutputNodeGene(key, config)
    gene2.aggregation = 'b'
    gene2.bias = 1
    return gene1, gene2


def get_simple_node_gene(key, config):
    """
    Create two output genes, one enabled with ('a', 'a', 0) and the other with ('b', 'b', 1) for (activation,
    aggregation, bias) respectively.
    """
    gene1 = SimpleNodeGene(key, config)
    gene1.activation = 'a'
    gene1.aggregation = 'a'
    gene1.bias = 0
    gene2 = SimpleNodeGene(key, config)
    gene2.activation = 'b'
    gene2.aggregation = 'b'
    gene2.bias = 1
    return gene1, gene2


class SimpleNode(unittest.TestCase):
    """Test the SimpleNodeGene's mutation operations."""
    
    def test_activation(self):
        """> Test activation received during crossover."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        cfg.genome.activation_default = 'a'
        cfg.genome.activation_options = {'a': 1, 'b': 2}
        gene1, gene2 = get_simple_node_gene(0, cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents
        p1 = False
        p2 = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            if gene3.activation == gene1.activation:
                p1 = True
            elif gene3.activation == gene2.activation:
                p2 = True
            else:
                raise self.failureException("Must be mutated to one of parent's values")
            if p1 and p2: break
        self.assertTrue(p1 and p2)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(gene3.activation, gene1.activation)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertEqual(gene3.activation, gene2.activation)
    
    def test_aggregation(self):
        """> Test aggregation received during crossover."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        cfg.genome.aggregation_default = 'a'
        cfg.genome.aggregation_options = {'a': 1, 'b': 2}
        gene1, gene2 = get_simple_node_gene(0, cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents
        p1 = False
        p2 = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            if gene3.aggregation == gene1.aggregation:
                p1 = True
            elif gene3.aggregation == gene2.aggregation:
                p2 = True
            else:
                raise self.failureException("Must be mutated to one of parent's values")
            if p1 and p2: break
        self.assertTrue(p1 and p2)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(gene3.aggregation, gene1.aggregation)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertEqual(gene3.aggregation, gene2.aggregation)
    
    def test_bias(self):
        """> Test bias received during crossover."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        cfg.genome.bias_min_value = -2
        cfg.genome.bias_max_value = 2
        gene1, gene2 = get_simple_node_gene(0, cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents
        p1 = False
        p2 = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            if gene3.bias == gene1.bias:
                p1 = True
            elif gene3.bias == gene2.bias:
                p2 = True
            else:
                raise self.failureException("Must be mutated to one of parent's values")
            if p1 and p2: break
        self.assertTrue(p1 and p2)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(gene3.bias, gene1.bias)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertEqual(gene3.bias, gene2.bias)
    
    def test_change(self):
        """> Test if a change to the created gene influences its parents."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents and deepcopy everything (just to be sure)
        cfg = Config().genome
        gene1, gene2 = get_simple_node_gene(0, cfg)
        gene1_act = deepcopy(gene1.activation)
        gene1_agg = deepcopy(gene1.aggregation)
        gene1_bias = deepcopy(gene1.bias)
        gene2_act = deepcopy(gene2.activation)
        gene2_agg = deepcopy(gene2.aggregation)
        gene2_bias = deepcopy(gene2.bias)
        
        # Perform crossover and mutations
        gene3 = gene1.crossover(other=gene2, cfg=cfg, ratio=0.5)
        gene3.activation = 'c'
        gene3.aggregation = 'c'
        gene3.bias = -10
        
        # Check for unchanged parents
        self.assertEqual(gene1.activation, gene1_act)
        self.assertEqual(gene1.aggregation, gene1_agg)
        self.assertEqual(gene1.bias, gene1_bias)
        self.assertEqual(gene2.activation, gene2_act)
        self.assertEqual(gene2.aggregation, gene2_agg)
        self.assertEqual(gene2.bias, gene2_bias)


class OutputNode(unittest.TestCase):
    """Test the OutputNodeGene's mutation operations."""
    
    def test_activation(self):
        """> Test activation remains unchanged during crossover."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        cfg.genome.activation_default = 'a'
        cfg.genome.activation_options = {'a': 1, 'b': 2}
        gene1, gene2 = get_output_node_gene(0, cfg.genome)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertEqual(gene3.activation, D_TANH)
    
    def test_aggregation(self):
        """> Test aggregation received during crossover."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        cfg.genome.aggregation_default = 'a'
        cfg.genome.aggregation_options = {'a': 1, 'b': 2}
        gene1, gene2 = get_output_node_gene(0, cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents
        p1 = False
        p2 = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            if gene3.aggregation == gene1.aggregation:
                p1 = True
            elif gene3.aggregation == gene2.aggregation:
                p2 = True
            else:
                raise self.failureException("Must be mutated to one of parent's values")
            if p1 and p2: break
        self.assertTrue(p1 and p2)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(gene3.aggregation, gene1.aggregation)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertEqual(gene3.aggregation, gene2.aggregation)
    
    def test_bias(self):
        """> Test bias received during crossover."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        cfg.genome.bias_min_value = -2
        cfg.genome.bias_max_value = 2
        gene1, gene2 = get_output_node_gene(0, cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents
        p1 = False
        p2 = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            if gene3.bias == gene1.bias:
                p1 = True
            elif gene3.bias == gene2.bias:
                p2 = True
            else:
                raise self.failureException("Must be mutated to one of parent's values")
            if p1 and p2: break
        self.assertTrue(p1 and p2)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(gene3.bias, gene1.bias)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertEqual(gene3.bias, gene2.bias)
    
    def test_change(self):
        """> Test if a change to the created gene influences its parents."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents and deepcopy everything (just to be sure)
        cfg = Config().genome
        gene1, gene2 = get_output_node_gene(0, cfg)
        gene1_act = deepcopy(gene1.activation)
        gene1_agg = deepcopy(gene1.aggregation)
        gene1_bias = deepcopy(gene1.bias)
        gene2_act = deepcopy(gene2.activation)
        gene2_agg = deepcopy(gene2.aggregation)
        gene2_bias = deepcopy(gene2.bias)
        
        # Perform crossover and mutations
        gene3 = gene1.crossover(other=gene2, cfg=cfg, ratio=0.5)
        gene3.activation = 'c'
        gene3.aggregation = 'c'
        gene3.bias = -10
        
        # Check for unchanged parents
        self.assertEqual(gene1.activation, gene1_act)
        self.assertEqual(gene1.aggregation, gene1_agg)
        self.assertEqual(gene1.bias, gene1_bias)
        self.assertEqual(gene2.activation, gene2_act)
        self.assertEqual(gene2.aggregation, gene2_agg)
        self.assertEqual(gene2.bias, gene2_bias)


class GruNode(unittest.TestCase):
    """Test the GruNodeGene's mutation operations."""
    
    def test_activation(self):
        """> Test activation received during crossover."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        cfg.genome.activation_default = 'a'
        cfg.genome.activation_options = {'a': 1, 'b': 2}
        gene1, gene2 = get_gru_node_gene(0, cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents
        p1 = False
        p2 = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            if gene3.activation == gene1.activation:
                p1 = True
            elif gene3.activation == gene2.activation:
                p2 = True
            else:
                raise self.failureException("Must be mutated to one of parent's values")
            if p1 and p2: break
        self.assertTrue(p1 and p2)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(gene3.activation, gene1.activation)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertEqual(gene3.activation, gene2.activation)
    
    def test_bias(self):
        """> Test if bias remains zero during crossover."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        gene1, gene2 = get_gru_node_gene(0, cfg.genome)
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            self.assertEqual(gene3.bias, 0)
    
    def test_bias_hh(self):
        """> Test if bias_hh is inherited from both parents."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        gene1, gene2 = get_gru_node_gene(0, cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents
        p1 = False
        p2 = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            for v in gene3.bias_hh:
                if v == 0:
                    p1 = True
                elif v == 1:
                    p2 = True
                else:
                    raise self.failureException("Must be mutated to one of parent's values")
            if p1 and p2: break
        self.assertTrue(p1 and p2)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(10):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(np.linalg.norm(gene3.bias_hh - gene1.bias_hh), 0)
            self.assertNotEqual(np.linalg.norm(gene3.bias_hh - gene2.bias_hh), 0)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(10):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertNotEqual(np.linalg.norm(gene3.bias_hh - gene1.bias_hh), 0)
            self.assertEqual(np.linalg.norm(gene3.bias_hh - gene2.bias_hh), 0)
    
    def test_bias_ih(self):
        """> Test if bias_ih is inherited from both parents."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        gene1, gene2 = get_gru_node_gene(0, cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents
        p1 = False
        p2 = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            for v in gene3.bias_ih:
                if v == 0:
                    p1 = True
                elif v == 1:
                    p2 = True
                else:
                    raise self.failureException("Must be mutated to one of parent's values")
            if p1 and p2: break
        self.assertTrue(p1 and p2)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(10):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(np.linalg.norm(gene3.bias_ih - gene1.bias_ih), 0)
            self.assertNotEqual(np.linalg.norm(gene3.bias_ih - gene2.bias_ih), 0)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(10):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertNotEqual(np.linalg.norm(gene3.bias_ih - gene1.bias_ih), 0)
            self.assertEqual(np.linalg.norm(gene3.bias_ih - gene2.bias_ih), 0)
    
    def test_weight_hh(self):
        """> Test if weight_hh is inherited from both parents."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        gene1, gene2 = get_gru_node_gene(0, cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents
        p1 = False
        p2 = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            for value in gene3.weight_hh:
                for v in value:
                    if v == 0:
                        p1 = True
                    elif v == 1:
                        p2 = True
                    else:
                        raise self.failureException("Must be mutated to one of parent's values")
            if p1 and p2: break
        self.assertTrue(p1 and p2)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(10):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(np.linalg.norm(gene3.weight_hh - gene1.weight_hh), 0)
            self.assertNotEqual(np.linalg.norm(gene3.weight_hh - gene2.weight_hh), 0)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(10):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertNotEqual(np.linalg.norm(gene3.weight_hh - gene1.weight_hh), 0)
            self.assertEqual(np.linalg.norm(gene3.weight_hh - gene2.weight_hh), 0)
    
    def test_weight_ih(self):
        """> Test if weight_ih is crossed properly by both parents."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        gene1, gene2 = get_gru_node_gene(0, cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents in shared column
        p1 = False
        p2 = False
        mixed = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            gene3.add_input_key(cfg=cfg.genome, k=-1)  # Implemented outside of gene's scope
            gene3.update_weight_ih()
            self.assertEqual(gene3.weight_ih.shape, (3, 1))
            
            # Check if shared column consists out of values found in both parents
            for value in gene3.weight_ih[:, 0]:
                if value == 0:
                    p1 = True
                elif value == 1:
                    p2 = True
                else:
                    raise self.failureException("weight_ih column must be inherited by one of both parents")
            
            # Mixed columns should exist
            if np.linalg.norm(gene3.weight_ih[:, 0] - gene1.weight_ih[:, 0]) != 0 and \
                    np.linalg.norm(gene3.weight_ih[:, 0] - gene2.weight_ih[:, 0]) != 0:
                mixed = True
            if p1 and p2 and mixed: break
        self.assertTrue(p1 and p2 and mixed)
        
        # Second column is only present in first gene, hence is fully copied
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            gene3.add_input_key(cfg=cfg.genome, k=-2)  # Implemented outside of gene's scope
            gene3.update_weight_ih()
            self.assertEqual(gene3.weight_ih.shape, (3, 1))
            self.assertEqual(np.linalg.norm(gene3.weight_ih[:, 0] - gene1.weight_ih[:, 0]), 0)
            self.assertNotEqual(np.linalg.norm(gene3.weight_ih[:, 0] - gene2.weight_ih[:, 0]), 0)
        
        # Third column is only present in second gene, hence is fully copied
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            gene3.add_input_key(cfg=cfg.genome, k=-3)  # Implemented outside of gene's scope
            gene3.update_weight_ih()
            self.assertEqual(gene3.weight_ih.shape, (3, 1))
            self.assertNotEqual(np.linalg.norm(gene3.weight_ih[:, 0] - gene1.weight_ih[:, 0]), 0)
            self.assertEqual(np.linalg.norm(gene3.weight_ih[:, 0] - gene2.weight_ih[:, 0]), 0)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(10):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            gene3.add_input_key(cfg=cfg.genome, k=-1)  # Implemented outside of gene's scope
            gene3.update_weight_ih()
            self.assertEqual(gene3.weight_ih.shape, (3, 1))
            self.assertEqual(np.linalg.norm(gene3.weight_ih[:, 0] - gene1.weight_ih[:, 0]), 0)
            self.assertNotEqual(np.linalg.norm(gene3.weight_ih[:, 0] - gene2.weight_ih[:, 0]), 0)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(10):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            gene3.add_input_key(cfg=cfg.genome, k=-1)  # Implemented outside of gene's scope
            gene3.update_weight_ih()
            self.assertEqual(gene3.weight_ih.shape, (3, 1))
            self.assertNotEqual(np.linalg.norm(gene3.weight_ih[:, 0] - gene1.weight_ih[:, 0]), 0)
            self.assertEqual(np.linalg.norm(gene3.weight_ih[:, 0] - gene2.weight_ih[:, 0]), 0)
    
    def test_weight_ih_full(self):
        """> Test if weight_ih_full is crossed properly by both parents."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        gene1, gene2 = get_gru_node_gene(0, cfg.genome)
        self.assertEqual(gene1.weight_ih_full.shape, (3, 2))
        self.assertEqual(gene2.weight_ih_full.shape, (3, 2))
        
        # Ratio of 0.5, so possible to cross to both parents in shared column, other columns are inherited from
        #  corresponding parent
        mixed = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            self.assertEqual(gene3.weight_ih_full.shape, (3, 3))
            
            # Check if column of the shared key (-1) is mixed (not always the case!)
            #  Due to sorted(keys), this is the last key
            if np.linalg.norm(gene3.weight_ih_full[:, 2] - gene1.weight_ih_full[:, 1]) != 0 and \
                    np.linalg.norm(gene3.weight_ih_full[:, 2] - gene2.weight_ih_full[:, 1]) != 0:
                mixed = True
            
            # Second column is always from first gene
            self.assertEqual(np.linalg.norm(gene3.weight_ih_full[:, 1] - gene1.weight_ih_full[:, 1]), 0)
            
            # Third column is always from second gene
            self.assertEqual(np.linalg.norm(gene3.weight_ih_full[:, 0] - gene2.weight_ih_full[:, 0]), 0)
            if mixed: break
        self.assertTrue(mixed)
        
        # Ratio of 1, so first column always inherits from first parent
        for _ in range(10):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(np.linalg.norm(gene3.weight_ih_full[:, 2] - gene1.weight_ih_full[:, 1]), 0)
            self.assertEqual(np.linalg.norm(gene3.weight_ih_full[:, 1] - gene1.weight_ih_full[:, 1]), 0)
            self.assertEqual(np.linalg.norm(gene3.weight_ih_full[:, 0] - gene2.weight_ih_full[:, 0]), 0)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(10):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertEqual(np.linalg.norm(gene3.weight_ih_full[:, 2] - gene2.weight_ih_full[:, 1]), 0)
            self.assertEqual(np.linalg.norm(gene3.weight_ih_full[:, 1] - gene1.weight_ih_full[:, 1]), 0)
            self.assertEqual(np.linalg.norm(gene3.weight_ih_full[:, 0] - gene2.weight_ih_full[:, 0]), 0)
    
    def test_change(self):
        """> Test if a change to the created gene influences its parents."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents and deepcopy everything (just to be sure)
        cfg = Config().genome
        gene1, gene2 = get_gru_node_gene(0, cfg)
        gene1_act = deepcopy(gene1.activation)
        gene1_bias = deepcopy(gene1.bias)
        gene1_bias_hh = deepcopy(gene1.bias_hh)
        gene1_bias_ih = deepcopy(gene1.bias_ih)
        gene1_weight_hh = deepcopy(gene1.weight_hh)
        gene1_weight_ih = deepcopy(gene1.weight_ih)
        gene1_weight_ih_full = deepcopy(gene1.weight_ih_full)
        gene2_act = deepcopy(gene2.activation)
        gene2_bias = deepcopy(gene2.bias)
        gene2_bias_hh = deepcopy(gene2.bias_hh)
        gene2_bias_ih = deepcopy(gene2.bias_ih)
        gene2_weight_hh = deepcopy(gene2.weight_hh)
        gene2_weight_ih = deepcopy(gene2.weight_ih)
        gene2_weight_ih_full = deepcopy(gene2.weight_ih_full)
        
        # Perform crossover and mutations
        gene3 = gene1.crossover(other=gene2, cfg=cfg, ratio=0.5)
        gene3.add_input_key(cfg=cfg, k=-1)
        gene3.update_weight_ih()
        gene3.activation = 'c'
        gene3.bias = -10
        gene3.bias_hh[0] = -10  # Make modifications directly on the vector
        gene3.bias_ih[0] = -10  # Make modifications directly on the vector
        gene3.weight_hh[0, 0] = -10  # Make modifications directly on the vector
        gene3.weight_ih[0, 0] = -10  # Make modifications directly on the vector
        gene3.weight_ih_full[0, 0] = -10  # Make modifications directly on the vector
        
        # Check for unchanged parents
        self.assertEqual(gene1.activation, gene1_act)
        self.assertEqual(gene1.bias, gene1_bias)
        self.assertEqual(np.linalg.norm(gene1.bias_hh - gene1_bias_hh), 0)
        self.assertEqual(np.linalg.norm(gene1.bias_ih - gene1_bias_ih), 0)
        self.assertEqual(np.linalg.norm(gene1.weight_hh - gene1_weight_hh), 0)
        self.assertEqual(np.linalg.norm(gene1.weight_ih - gene1_weight_ih), 0)
        self.assertEqual(np.linalg.norm(gene1.weight_ih_full - gene1_weight_ih_full), 0)
        self.assertEqual(gene2.activation, gene2_act)
        self.assertEqual(gene2.bias, gene2_bias)
        self.assertEqual(np.linalg.norm(gene2.bias_hh - gene2_bias_hh), 0)
        self.assertEqual(np.linalg.norm(gene2.bias_ih - gene2_bias_ih), 0)
        self.assertEqual(np.linalg.norm(gene2.weight_hh - gene2_weight_hh), 0)
        self.assertEqual(np.linalg.norm(gene2.weight_ih - gene2_weight_ih), 0)
        self.assertEqual(np.linalg.norm(gene2.weight_ih_full - gene2_weight_ih_full), 0)


class Connection(unittest.TestCase):
    """Test the ConnectionGene's mutation operations."""
    
    def test_enabled(self):
        """> Test if enabled is crossed correctly."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        gene1, gene2 = get_connection_genes((-1, 0), cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents
        p1 = False
        p2 = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            if gene3.enabled == gene1.enabled:
                p1 = True
            elif gene3.enabled == gene2.enabled:
                p2 = True
            else:
                raise self.failureException("Must be mutated to one of parent's values")
            if p1 and p2: break
        self.assertTrue(p1 and p2)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(gene3.enabled, gene1.enabled)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertEqual(gene3.enabled, gene2.enabled)
    
    def test_weight(self):
        """> Test weight received during crossover."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents
        cfg = Config()
        cfg.genome.weight_min_value = -2
        cfg.genome.weight_max_value = 2
        gene1, gene2 = get_connection_genes((-1, 0), cfg.genome)
        
        # Ratio of 0.5, so possible to cross to both parents
        p1 = False
        p2 = False
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0.5)
            if gene3.weight == gene1.weight:
                p1 = True
            elif gene3.weight == gene2.weight:
                p2 = True
            else:
                raise self.failureException("Must be mutated to one of parent's values")
            if p1 and p2: break
        self.assertTrue(p1 and p2)
        
        # Ratio of 1, so always inherits from first parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=1)
            self.assertEqual(gene3.weight, gene1.weight)
        
        # Ratio of 0, so always inherits from second parent
        for _ in range(100):
            gene3 = gene1.crossover(other=gene2, cfg=cfg.genome, ratio=0)
            self.assertEqual(gene3.weight, gene2.weight)
    
    def test_change(self):
        """> Test if a change to the created gene influences its parents."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create parents and deepcopy everything (just to be sure)
        cfg = Config().genome
        gene1, gene2 = get_connection_genes((-1, 0), cfg)
        gene1_en = deepcopy(gene1.enabled)
        gene1_w = deepcopy(gene1.weight)
        gene2_en = deepcopy(gene2.enabled)
        gene2_w = deepcopy(gene2.weight)
        
        # Perform crossover and mutations
        gene3 = gene1.crossover(other=gene2, cfg=cfg, ratio=0.5)
        gene3.enabled = False
        gene3.weight = -10
        
        # Check for unchanged parents
        self.assertEqual(gene1.enabled, gene1_en)
        self.assertEqual(gene1.weight, gene1_w)
        self.assertEqual(gene2.enabled, gene2_en)
        self.assertEqual(gene2.weight, gene2_w)


def main():
    # Test the SimpleNodeGene
    sn = SimpleNode()
    sn.test_activation()
    sn.test_aggregation()
    sn.test_bias()
    sn.test_change()
    
    # Test the OutputNodeGene
    on = OutputNode()
    on.test_activation()
    on.test_aggregation()
    on.test_bias()
    on.test_change()
    
    # Test the GruNodeGene
    gn = GruNode()
    gn.test_activation()
    gn.test_bias()
    gn.test_bias_hh()
    gn.test_bias_ih()
    gn.test_weight_hh()
    gn.test_weight_ih()
    gn.test_weight_ih_full()
    gn.test_change()
    
    # Test the ConnectionGene
    c = Connection()
    c.test_enabled()
    c.test_weight()
    c.test_change()


if __name__ == '__main__':
    unittest.main()
