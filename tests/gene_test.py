"""
genome_test.py

Test gene-specific operations.
"""
import os
import unittest

from population.utils.genome_util.genes import SimpleNodeGene


class SimpleNode(unittest.TestCase):
    """Test connection-mutation mechanism in the genomes."""
    
    def test_mutation(self):
        """> TODO"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        
        pass  # Dummy


def get_simple_gene(key, config):
    """Create a random initialized gene."""
    gene = SimpleNodeGene(key)
    gene.init_attributes(config)
    return gene


def get_config():
    """Get a modified config-file."""


def main():
    # Test wall collision
    sn = SimpleNode()


if __name__ == '__main__':
    unittest.main()
