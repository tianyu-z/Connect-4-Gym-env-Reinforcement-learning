from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import BabyPlayer
from stable_baselines3 import PPO, DQN
from connect_four_gymnasium.players.ModelPlayer import ModelPlayer
from connect_four_gymnasium.tools import EloLeaderboard
from net import CustomAZPolicy
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os
from tqdm import tqdm
import numpy as np

fix_seed = True


if fix_seed:
    SEED = 42
    # Set the seed for PyTorch
    torch.manual_seed(SEED)
    # Set the seed for NumPy
    np.random.seed(SEED)
verbose = 1
# model = PPO(CustomAZPolicy, env, n_steps=131072, batch_size=4096, verbose=1) # init training
model1 = DQN.load("C:\\Users\\tiany\\Documents\\best_model_DQN.zip")  # resume training
model2 = PPO.load(
    "C:\\Users\\tiany\\Documents\\best_model_PPO_selfplay.zip"
)  # resume training


# Set the environment for the model
myself = ModelPlayer(model1, name="DQN", deteministic=False)
opponent = ModelPlayer(model2, name="PPO_adult", deteministic=False)
render_mode = "human" if verbose >= 2 else "rgb_array"


def play_once(verbose=verbose, first="DQN"):
    if first == "DQN":
        env = ConnectFourEnv(
            opponent=opponent,
            render_mode=render_mode,
            first_player=1,
            main_player_name="DQN",
        )
    else:
        env = ConnectFourEnv(
            opponent=opponent,
            render_mode=render_mode,
            first_player=-1,
            main_player_name="DQN",
        )
    model1.set_env(env)
    obs, done = env.reset()
    while True:
        actions = myself.play(obs)
        obs, reward, done, truncated, info = env.step(actions)
        if done:
            break
    return reward


if __name__ == "__main__":
    result = {"win": 0, "lose": 0, "draw": 0}
    for i in tqdm(range(1000)):
        isAFNwin = play_once(first="PPO")
        if isAFNwin == 1:
            result["win"] += 1
        elif isAFNwin == -1:
            result["lose"] += 1
        elif isAFNwin == 0:
            result["draw"] += 1
        else:
            raise Exception("Invalid result!")
    print(result)
