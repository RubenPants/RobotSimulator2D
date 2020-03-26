"""
genome_crossover_test.py

TODO: Finish test-cases!

Test genome- and inter-genome-specific operations.
"""
import os
import unittest


class ConnectionMutation(unittest.TestCase):
    """Test connection-mutation mechanism in the genomes."""
    
    def test_something(self):
        """> TODO"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        pass  # Dummy


def create_genome():
    """TODO"""


def main():
    # Test wall collision
    gwc = ConnectionMutation()


if __name__ == '__main__':
    unittest.main()
