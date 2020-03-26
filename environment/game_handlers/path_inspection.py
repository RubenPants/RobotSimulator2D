"""
path_inspection.py

Check how long it takes to finish a maze on average.
"""
import os
from collections import Counter
import matplotlib.pyplot as plt

from tqdm import tqdm

from config import Config
from environment.entities.game import D_A_STAR, Game

if __name__ == '__main__':
    os.chdir("../..")
    config = Config()
    counter = Counter()
    for g_id in tqdm(range(1, config.game.max_eval_game_id + 1)):
        game = Game(
                game_id=g_id,
                config=config,
                save_path="environment/games_db/",
                overwrite=False,
                silent=True,
        )
        a_star = game.game_params()[D_A_STAR]
        counter[round(a_star / game.bot_config.driving_speed)] += 1
    
    # Visualize the counter
    x, y = zip(*sorted(counter.items()))
    plt.figure()
    plt.bar(x, y)
    plt.title("Minimum time it takes to reach the target")
    plt.xlabel("time (s)")
    plt.ylabel("Number of games")
    plt.tight_layout()
    plt.savefig('environment/visualizations/a_star.png')
    plt.show()
    plt.close()
