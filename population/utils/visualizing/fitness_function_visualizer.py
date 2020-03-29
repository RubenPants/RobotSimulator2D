"""
fitness_functions.py

Create visualizations for each of the fitness functions.
"""
import argparse
import os
from math import sqrt

import matplotlib as mpl
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import clip

from config import GameConfig
from environment.entities.game import get_game
from utils.dictionary import *


def distance(save: bool = True):
    """Create an image for the distance-fitness."""
    cfg = GameConfig()
    diagonal = sqrt(cfg.x_axis ** 2 + cfg.y_axis ** 2)
    
    # Define the function
    def get_score(d, reached=False):
        """Get a score for the given distance."""
        return 1 if reached else clip((1 - (d - cfg.target_reached) / diagonal) ** 2, a_min=0, a_max=1)
    
    # Create the figure
    x = []
    y = []
    for i in range(0, round(diagonal * 100)):
        x.append(i / 100)
        y.append(get_score(i / 100))
    
    plt.figure()
    
    # Plot the distance function
    plt.plot(x, y, 'b', label='distance-based score')
    plt.axvspan(0, cfg.target_reached, alpha=0.5, color='green', label='target reached')
    
    # Beautify the plot
    plt.title('Fitness in function of distance to target')
    plt.xlabel("Distance to target")
    plt.xticks([i * 2 for i in range(round(diagonal / 2) + 1)])
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='x')
    if save: plt.savefig('population/utils/visualizing/images/distance_fitness.png')
    plt.show()
    plt.close()


def path(game_id: int, save: bool = True):
    """Create an image for the path-fitness."""
    cfg = GameConfig()
    game = get_game(i=game_id)  # Dummy game
    params = game.game_params()
    path = params[D_PATH]
    a_star = params[D_A_STAR]
    
    # Define the function
    def get_score(p):
        """Get a score for the given path-position."""
        temp = path[round(p[0], 1), round(p[1], 1)] / a_star
        return (clip(1 - temp, a_min=0, a_max=1) + clip(1 - temp, a_min=0, a_max=1) ** 2) / 2
    
    # Create the figure
    score = dict()
    for x in range(0, cfg.x_axis * 10 + 1):
        for y in range(0, cfg.x_axis * 10 + 1):
            score[x / 10, y / 10] = get_score((x / 10, y / 10))
    
    # Create the figure
    def fill(x1, x2, y1, y2):
        c = clr.to_hex([1, score[(x2, y2)], 0])
        plt.fill([x1, x1, x2, x2], [y1, y2, y2, y1], c)
    
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    game.get_blueprint(ax)
    for x in range(0, 140):
        for y in range(0, 140):
            fill(round(x / 10, 1), round((x + 1) / 10, 1), round(y / 10, 1), round((y + 1) / 10, 1))
    game.get_blueprint(ax)
    plt.title(f"Path-score by position for game {game.id:05d}")
    cax = divider.append_axes('right', size='5%', pad=0.05)  # Create the colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    data = np.ones((256, 3))
    data[:, 1] = np.linspace(0.33, 1, 256)
    data[:, 2] = 0
    mpl.colorbar.ColorbarBase(cax, cmap=clr.ListedColormap(data), norm=norm, orientation='vertical')
    plt.tight_layout()
    if save: plt.savefig(f'population/utils/visualizing/images/path_fitness_game_{game.id:05d}.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save', type=bool, default=True)
    
    # Fitness functions
    parser.add_argument('--distance', type=bool, default=True)
    parser.add_argument('--path', type=bool, default=True)
    args = parser.parse_args()
    
    # Go back to root
    os.chdir('../../../')
    
    if args.distance:
        distance(
                save=args.save
        )
    if args.path:
        path(
                game_id=0,
                save=args.save
        )
