import sys

sys.path.append("../")
from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import SelfTrained6Player, ConsolePlayer
from stable_baselines3 import PPO
from connect_four_gymnasium.players.ModelPlayer import ModelPlayer
from ours.our_env import Connect4Env, get_env_dims, BaseEnv, Player, Outcome
from ours.our_models import AZResNet
import os
import torch
import pickle
import numpy as np
import multiprocessing
from tqdm import tqdm


fix_seed = True
verbose = 1


if fix_seed:
    SEED = 42
    # Set the seed for PyTorch
    torch.manual_seed(SEED)
    # Set the seed for NumPy
    np.random.seed(SEED)


# Load or train your PPO model
PPO_model_path = "C:\\Users\\tiany\\Documents\\best_model_PPO_vs_AdultSmarterPlayer.zip"  # Specify the path to your trained model
model = PPO.load(PPO_model_path)
NN = ModelPlayer(model, name="PPO", deteministic=False)

# Initialize players
AFN_connect4 = Connect4Env()
state_dims, action_dims = get_env_dims(AFN_connect4)


# Initialize the AFN
AFN = AZResNet(state_dims, action_dims)
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
AFN_model_path = "H:\\vscode_git\\GFlowChess\\tb\\pt\\afn_Connect4_15_29_state_dict.pth"

if AFN_model_path is not None:
    if not os.path.isabs(AFN_model_path):
        # Construct the absolute path to the model file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        AFN_model_path = os.path.join(script_dir, AFN_model_path)
    if AFN_model_path.endswith(".pth") or AFN_model_path.endswith(".pt"):
        AFN.load_state_dict(
            torch.load(AFN_model_path, map_location=torch.device("cpu"))
        )
    elif AFN_model_path.endswith(".pkl"):
        with open(AFN_model_path, "rb") as f:
            data = pickle.load(f)
            AFN = data["model"]
AFN.to(DEVICE)
AFN.eval()


def play_once(verbose=verbose, first="AFN"):
    if first == "AFN":
        # Initialize the environment
        render_mode = "human" if verbose >= 2 else "rgb_array"
        PPO_connect4 = ConnectFourEnv(
            opponent=NN, render_mode=render_mode, first_player=1, main_player_name="AFN"
        )
        if fix_seed:
            PPO_connect4.seed(SEED)
        # Game loop
        obs, _ = PPO_connect4.reset()
        AFN_connect4.reset()
        while True:
            AFN_action = AFN.sample_actions(AFN_connect4, 0).item()
            obs, done = AFN_connect4.step(AFN_action)
            if done:
                break
            obs, rewards, done, truncated, info = PPO_connect4.step(AFN_action)
            PPO_action = info["opponent_action"]
            if done:
                break
            obs, done = AFN_connect4.step(PPO_action)
            if done:
                break
        if AFN_connect4.outcome == Outcome.WIN_P1:
            if verbose:
                print("AFN won!")
            isAFNwin = 1
        elif AFN_connect4.outcome == Outcome.WIN_P2:
            if verbose:
                print("PPO won!")
            isAFNwin = -1
        else:
            if verbose:
                print("It's a draw!")
            isAFNwin = 0
        return isAFNwin
    else:
        # Initialize the environment
        render_mode = "human" if verbose >= 2 else "rgb_array"
        PPO_connect4 = ConnectFourEnv(
            opponent=NN,
            render_mode=render_mode,
            first_player=-1,
            main_player_name="AFN",
        )
        if fix_seed:
            PPO_connect4.seed(SEED)
        # Game loop
        obs, info = PPO_connect4.reset()
        PPO_action = info["opponent_action"]
        AFN_connect4.reset()
        obs, done = AFN_connect4.step(PPO_action)
        while True:
            AFN_action = AFN.sample_actions(AFN_connect4, 0).item()
            obs, done = AFN_connect4.step(AFN_action)
            if done:
                break
            obs, rewards, done, truncated, info = PPO_connect4.step(AFN_action)
            PPO_action = info["opponent_action"]
            if done:
                break
            obs, done = AFN_connect4.step(PPO_action)
            if done:
                break
        if AFN_connect4.outcome == Outcome.WIN_P1:
            if verbose:
                print("PPO won!")
            isAFNwin = -1
        elif AFN_connect4.outcome == Outcome.WIN_P2:
            if verbose:
                print("AFN won!")
            isAFNwin = 1
        else:
            if verbose:
                print("It's a draw!")
            isAFNwin = 0
        return isAFNwin


def play_once_wrapper(_):
    try:
        return play_once()
    except Exception as e:
        # Optionally, log the error
        # print(f"An error occurred: {e}")
        return None  # Special value indicating error


if __name__ == "__main__":
    result = {"win": 0, "lose": 0, "draw": 0}
    num_processes = multiprocessing.cpu_count()  # or set a specific number

    with multiprocessing.Pool(num_processes) as pool:
        # Using tqdm to create a progress bar
        results = list(tqdm(pool.imap(play_once_wrapper, range(1000)), total=1000))

    for isAFNwin in results:
        if isAFNwin == 1:
            result["win"] += 1
        elif isAFNwin == -1:
            result["lose"] += 1
        elif isAFNwin == 0:
            result["draw"] += 1
        else:
            raise Exception("Invalid result!")
    print(result)
