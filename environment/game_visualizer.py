"""
game_visualizer.py

This method is purely used to visualize the game-maps (used in the thesis). This file must be called in the environment
folder to make sure the visualizations are saved properly.
"""
import matplotlib as mpl
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib import cm
from environment.entities.game import Game


def load_game(game_id):
    """
    Load the game of the given ID.
    
    :param game_id: Integer
    :return: Game object
    """
    return Game(game_id=game_id,
                rel_path="",
                silent=False)


def game_blueprint(game):
    """
    Create the blueprint of the game.
    
    :param game: Game object
    """
    game.get_blueprint()
    plt.title("Blueprint - Game {id:05d}".format(id=game.id))
    plt.savefig('visualizations/blueprint_game{g:05d}'.format(g=game.id))
    plt.close()


def path_heatmap(game):
    path = game.path
    path_norm = dict((k, v / max(path.values())) for k, v in path.items())
    
    def fill(x1, x2, y1, y2):
        x12 = (x1 + x2) / 2
        y12 = (y1 + y2) / 2
        c = clr.to_hex([1, path_norm[(x12, y12)] * 2 / 3 + 0.33, 0])
        plt.fill([x1, x1, x2, x2], [y1, y2, y2, y1], c)
    
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    game.get_blueprint(ax)
    for x in range(14):
        for y in range(14):
            fill(x, x + 1, y, y + 1)
    game.get_blueprint(ax)
    
    # Set the title for the plot
    plt.title("Heatmap - Game {id:05d}".format(id=game.id))
    
    # Create the colorbar
    cax = divider.append_axes('right', size='5%', pad=0.05)
    norm = mpl.colors.Normalize(vmin=0, vmax=max(path.values()))
    data = np.ones((256, 3))
    data[:, 1] = np.linspace(0.33, 1, 256)
    data[:, 2] = 0
    mpl.colorbar.ColorbarBase(cax, cmap=clr.ListedColormap(data), norm=norm, orientation='vertical')
    
    # Save the figure
    plt.savefig('visualizations/heatmap_game{g:05d}'.format(g=game.id))


if __name__ == '__main__':
    # Load the game
    for g_id in range(1, 11):
        g = load_game(g_id)
        
        # Create visualizations
        game_blueprint(g)
        path_heatmap(g)
