import sys

sys.path.append("../")
from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import SelfTrained7Player, ConsolePlayer
from connect_four_gymnasium.players.ModelPlayer import ModelPlayer
from ours.our_env import Connect4Env, get_env_dims, BaseEnv, Player, Outcome
from ours.our_models import AZResNet
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm

fix_seed = True
verbose = 0


if fix_seed:
    SEED = 42
    # Set the seed for PyTorch
    torch.manual_seed(SEED)
    # Set the seed for NumPy
    np.random.seed(SEED)

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
        AZ_connect4 = ConnectFourEnv(
            opponent=SelfTrained7Player(deteministic=True),
            render_mode=render_mode,
            first_player=1,
            main_player_name="you",
        )
        if fix_seed:
            AZ_connect4.seed(SEED)
        # Game loop
        obs, _ = AZ_connect4.reset()
        AFN_connect4.reset()
        while True:
            AFN_action = AFN.sample_actions(AFN_connect4, 0).item()
            obs, done = AFN_connect4.step(AFN_action)
            if done:
                break
            obs, rewards, done, truncated, info = AZ_connect4.step(AFN_action)
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
        AZ_connect4 = ConnectFourEnv(
            opponent=SelfTrained7Player(deteministic=True),
            render_mode=render_mode,
            first_player=-1,
            main_player_name="you",
        )
        if fix_seed:
            AZ_connect4.seed(SEED)
        # Game loop
        obs, info = AZ_connect4.reset()
        PPO_action = info["opponent_action"]
        AFN_connect4.reset()
        obs, done = AFN_connect4.step(PPO_action)
        while True:
            AFN_action = AFN.sample_actions(AFN_connect4, 0).item()
            obs, done = AFN_connect4.step(AFN_action)
            if done:
                break
            obs, rewards, done, truncated, info = AZ_connect4.step(AFN_action)
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


if __name__ == "__main__":
    result = {"win": 0, "lose": 0, "draw": 0}
    for i in tqdm(range(1000)):
        isAFNwin = play_once(first="AlphaFour")
        if isAFNwin == 1:
            result["win"] += 1
        elif isAFNwin == -1:
            result["lose"] += 1
        elif isAFNwin == 0:
            result["draw"] += 1
        else:
            raise Exception("Invalid result!")
    print(result)
