"""
distance.py

Visualization for the distance fitness-function.
"""
import os

import matplotlib as mpl
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import clip

from config import GameConfig
from environment.entities.game import get_game
from utils.dictionary import *

# Go back to root
os.chdir('../../../../')

# Load in needed objects
cfg = GameConfig()
game = get_game(i=1)  # Dummy game
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
plt.savefig(f'population/utils/visualizing/fitness_visualizations/images/path_fitness_game_{game.id:05d}.png')
plt.show()
plt.close()
