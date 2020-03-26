"""
population_test.py

Test the basic functionality of the population.

:note: This test-suite will not disable prints of the population, hence it can be quite a mess...
"""
import os
import shutil
import unittest

from config import Config
from main import blueprint, evaluate, trace, trace_most_fit, train, train_same_games, visualize_genome
from population.population import Population


def get_population():
    """Get a dummy population with minimal configuration."""
    cfg = Config()
    cfg.game.batch = 1  # Only one game will be evaluated to improve performance
    cfg.game.duration = 10  # Small duration
    cfg.game.max_game_id = 10  # Only first 10 games can be used for training
    cfg.game.max_eval_game_id = 21  # Only the next game can be used for evaluation
    cfg.game.fps = 10  # Games of low accuracy but faster
    cfg.population.pop_size = 10  # Keep a small population
    cfg.population.compatibility_thr = 100.0  # Make sure that population does not expand
    
    pop = Population(
            name='delete_me',
            folder_name='test_scenario',
            config=cfg,
            log_print=False,
    )
    
    return pop, cfg


class PopulationTest(unittest.TestCase):
    """Test the basic population operations."""
    
    def setUp(self):
        """> Create the population used during testing."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create the population
        get_population()
    
    def test_creation(self):
        """> Test if population can be successfully created."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        self.assertTrue(os.path.exists('population/storage/test_scenario/delete_me/'))
    
    def test_train(self):
        """> Test if population can be successfully trained."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        train(
                population=pop,
                game_config=cfg,
                unused_cpu=0,
                iterations=1,
                debug=True,
        )
    
    def test_train_same(self):
        """> Test if population can be successfully trained in the same-game training task."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        train_same_games(
                population=pop,
                games=[1],  # Random game
                game_config=cfg,
                unused_cpu=0,
                iterations=1,
                debug=True,
        )
    
    def test_blueprint(self):
        """> Test if population can blueprint its results."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        blueprint(
                population=pop,
                games=[1],  # Random game
                game_config=cfg,
        )
    
    def test_trace(self):
        """> Test if population can trace its current population."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        trace(
                population=pop,
                games=[1],  # Random game
                game_config=cfg,
        )
    
    def test_trace_fit(self):
        """> Test if population can trace its best genome."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        genome = pop.best_genome if pop.best_genome else list(pop.population.values())[-1]
        trace_most_fit(
                population=pop,
                genome=genome,
                games=[1],  # Random game
                game_config=cfg,
        )
    
    def test_evaluate(self):
        """> Test if the population can be evaluated."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        evaluate(
                population=pop,
                game_config=cfg,
        )
    
    def test_genome_visualization(self):
        """> Test if a genome from the population can be visualized."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        genome = pop.best_genome if pop.best_genome else list(pop.population.values())[-1]
        visualize_genome(
                population=pop,
                genome=genome,
                debug=True,
                show=False,
        )
    
    def tearDown(self):
        """> Remove the population that was used for testing."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        shutil.rmtree('population/storage/test_scenario')


def main():
    # Test wall collision
    sn = PopulationTest()
    sn.test_creation()


if __name__ == '__main__':
    unittest.main()
