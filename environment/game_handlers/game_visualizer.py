"""
game_visualizer.py

This method is purely used to visualize the game-maps (used in the thesis). This file must be called in the environment
folder to make sure the visualizations are saved properly.
"""
import os
from math import cos, sin

import matplotlib as mpl
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from config import Config
from environment.entities.game import get_game


def game_blueprint(game):
    """
    Create the blueprint of the game.
    
    :param game: Game object
    """
    # Get game's blueprint
    game.get_blueprint()
    
    # Add arrow to indicate initial direction of robot
    x = game.player.init_pos[0]
    y = game.player.init_pos[1]
    dx = cos(game.player.init_angle)
    dy = sin(game.player.init_angle)
    plt.arrow(x, y, dx, dy, head_width=0.1, length_includes_head=True)
    
    plt.title("Blueprint - Game {id:05d}".format(id=game.id))
    plt.savefig(f'environment/visualizations/blueprint_game{game.id:05d}')
    plt.close()


def path_heatmap(game):
    path = game.path
    max_dist = max(path.values())
    path_norm = dict((k, v / max_dist) for k, v in path.items())
    
    def fill(x1, x2, y1, y2):
        c = clr.to_hex([1, path_norm[(x2, y2)] * 2 / 3 + 0.33, 0])
        plt.fill([x1, x1, x2, x2], [y1, y2, y2, y1], c)
    
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    game.get_blueprint(ax)
    for x in range(0, 140):
        for y in range(0, 140):
            fill(round(x / 10, 1), round((x + 1) / 10, 1), round(y / 10, 1), round((y + 1) / 10, 1))
    game.get_blueprint(ax)
    
    # Set the title for the plot
    plt.title("Heatmap - Game {id:05d}".format(id=game.id))
    
    # Create the colorbar
    cax = divider.append_axes('right', size='5%', pad=0.05)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_dist)
    data = np.ones((256, 3))
    data[:, 1] = np.linspace(0.33, 1, 256)
    data[:, 2] = 0
    mpl.colorbar.ColorbarBase(cax, cmap=clr.ListedColormap(data), norm=norm, orientation='vertical')
    
    # Save the figure
    plt.savefig(f'environment/visualizations/heatmap_game{game.id:05d}')


if __name__ == '__main__':
    os.chdir("../..")
    cfg = Config()
    # for g_id in [0]:
    for g_id in tqdm(range(1, 11)):
        # Load the game
        g = get_game(g_id, cfg=cfg)
        
        # Create visualizations
        game_blueprint(g)
        # path_heatmap(g)
