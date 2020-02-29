import os

from tqdm import tqdm

from configs.config import GameConfig
from environment.entities.game import Game

if __name__ == '__main__':
    os.chdir("..")
    config = GameConfig()
    for g_id in tqdm(range(1, config.max_eval_game_id + 1)):
        try:
            game = Game(
                    game_id=g_id,
                    save_path="environment/games_db/",
                    overwrite=False,
                    silent=True,
            )
            game.close()
            game.reset()
            game.get_blueprint()
            game.get_observation()
            game.step(0, 0)
        except Exception:
            print(f"Bug in game: {g_id}, please manually redo this one")
