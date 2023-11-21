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

# Load or train your PPO model
PPO_model_path = (
    "eval_logs/best_model_PPO.zip"  # Specify the path to your trained model
)
model = PPO.load(PPO_model_path)
NN = ModelPlayer(model, name="PPO")
# Initialize players
AFN_connect4 = Connect4Env()
state_dims, action_dims = get_env_dims(AFN_connect4)

# Initialize the environment
PPO_connect4 = ConnectFourEnv(opponent=NN, render_mode="human", main_player_name="AFN")

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

# Game loop
obs, _ = PPO_connect4.reset()
AFN_connect4.reset()
AFN_first = False
AFN_connect4.render()
while not AFN_connect4.done:
    if AFN_connect4.curr_player == Player.ONE:
        if AFN_first:
            AFN_action = AFN.sample_actions(AFN_connect4, 0).item()
            AFN_connect4.step(AFN_action)
            AFN_connect4.render()
        else:
            try:
                obs, rewards, dones, truncated, info = PPO_connect4.step(AFN_action)
            except:
                AFN_action = None
                obs, rewards, dones, truncated, info = PPO_connect4.step(AFN_action)
            valid_PPO_moves = np.where(AFN_connect4.get_masks())[0]
            PPO_action = info["oppoent_action"]
            assert PPO_action in valid_PPO_moves, "Invalid move! Please try again."
            AFN_connect4.step(PPO_action)
            AFN_connect4.render()
            PPO_connect4.render()
            if truncated or dones:
                obs, _ = PPO_connect4.reset()
                AFN_connect4.reset()
                AFN_connect4.render()
    else:
        if AFN_first:
            try:
                obs, rewards, dones, truncated, info = PPO_connect4.step(AFN_action)
            except:
                AFN_action = None
                obs, rewards, dones, truncated, info = PPO_connect4.step(AFN_action)
            valid_PPO_moves = np.where(AFN_connect4.get_masks())[0]
            PPO_action = info["oppoent_action"]
            assert PPO_action in valid_PPO_moves, "Invalid move! Please try again."
            AFN_connect4.step(PPO_action)
            AFN_connect4.render()
            PPO_connect4.render()
            if truncated or dones:
                obs, _ = PPO_connect4.reset()
                AFN_connect4.reset()
                AFN_connect4.render()
        else:
            AFN_action = AFN.sample_actions(AFN_connect4, 0).item()
            AFN_connect4.step(AFN_action)
            AFN_connect4.render()
AFN_connect4.render()
if not AFN_first:
    if AFN_connect4.outcome == Outcome.WIN_P2:
        print("AFN won!")
    elif AFN_connect4.outcome == Outcome.WIN_P1:
        print("PPO won!")
    else:
        print("It's a draw!")
else:
    if AFN_connect4.outcome == Outcome.WIN_P1:
        print("AFN won!")
    elif AFN_connect4.outcome == Outcome.WIN_P2:
        print("PPO won!")
    else:
        print("It's a draw!")
