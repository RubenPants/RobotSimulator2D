"""
inspect_games.py

Check if all games are created correctly.
"""
import os

from tqdm import tqdm

from config import Config
from environment.entities.game import Game, D_PATH

if __name__ == '__main__':
    os.chdir("../..")
    config = Config()
    for g_id in tqdm(range(1, config.game.max_eval_game_id + 1)):
        try:
            game = Game(
                    game_id=g_id,
                    config=config,
                    save_path="environment/games_db/",
                    overwrite=False,
                    silent=True,
            )
            game.close()
            game.reset()
            game.get_blueprint()
            game.get_observation()
            game.step(0, 0)
            path = game.game_params()[D_PATH]
            for i in range(0, 140 + 1):
                for j in range(0, 140 + 1):
                    key = (i / 10, j / 10)
                    if key not in path:
                        raise Exception("Key in path missing")
        except Exception:
            print(f"Bug in game: {g_id}, please manually redo this one")
