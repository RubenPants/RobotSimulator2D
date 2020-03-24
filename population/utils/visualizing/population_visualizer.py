"""
population_visualizer.py

Visualize the behaviour of a complete population.
"""
import matplotlib.pyplot as plt

from utils.dictionary import D_GAME_ID, D_POS
from utils.myutils import get_subfolder


def create_blueprints(final_observations: dict, games: list, gen: int, save_path: str):
    """
    Save images in the relative 'images/' subfolder of the population.

    :param final_observations: Dictionary of all the final game observations made
    :param games: List Game-objects used during evaluation
    :param gen: Population's current generation
    :param save_path: Path of 'images'-folder under which image must be saved
    """
    genome_keys = list(final_observations.keys())
    for g in games:
        # Get the game's blueprint
        g.get_blueprint()
        
        # Get all the final positions of the agents
        positions = []
        for gk in genome_keys:
            positions += [fo[D_POS] for fo in final_observations[gk] if fo[D_GAME_ID] == g.id]
        
        # Plot the positions
        dot_x = [p[0] for p in positions]
        dot_y = [p[1] for p in positions]
        plt.plot(dot_x, dot_y, 'ro')
        
        # Add title
        plt.title(f"Blueprint - Game {g.id:05d} - Generation {gen:05d}")
        
        # Save figure
        game_path = get_subfolder(save_path, 'game{id:05d}'.format(id=g.id))
        plt.savefig(f'{game_path}blueprint_gen{gen:05d}')
        plt.close()


def create_traces(traces: dict, games: list, gen: int, save_path: str, save_name: str = 'trace'):
    """
    Save images in the relative 'images/' subfolder of the population.

    :param traces: Dictionary of all the traces
    :param games: List Game-objects used during evaluation
    :param gen: Population's current generation
    :param save_path: Path of 'images'-folder under which image must be saved
    :param save_name: Name of saved file
    """
    genome_keys = list(traces.keys())
    for i, g in enumerate(games):
        # Get the game's blueprint
        g.get_blueprint()
        
        # Append the traces agent by agent
        for gk in genome_keys:
            # Get the trace of the genome for the requested game
            x_pos, y_pos = zip(*traces[gk][i])
            
            # Plot the trace (gradient)
            size = len(x_pos)
            for p in range(0, size - g.fps, g.fps):  # Trace each second of the run
                plt.plot((x_pos[p], x_pos[p + g.fps]), (y_pos[p], y_pos[p + g.fps]), color=(1, p / (1 * size), 0))
        
        # Add title
        plt.title(f"Traces - Game {g.id:05d} - Generation {gen:05d}")
        
        # Save figure
        game_path = get_subfolder(save_path, 'game{id:05d}'.format(id=g.id))
        plt.savefig(f'{game_path}{save_name}_gen{gen:05d}')
        plt.close()
