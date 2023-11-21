import sys

sys.path.append("../")
from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import SelfTrained6Player, ConsolePlayer
from stable_baselines3 import PPO
from connect_four_gymnasium.players.ModelPlayer import ModelPlayer

# Load or train your PPO model
model_path = "eval_logs/best_model_PPO.zip"  # Specify the path to your trained model
model = PPO.load(model_path)
NN = ModelPlayer(model, name="PPO")
# Initialize players
you = ConsolePlayer()

# Initialize the environment
env = ConnectFourEnv(opponent=NN, render_mode="human", main_player_name="human")

# Game loop
obs, _ = env.reset()
for i in range(5000):
    action = you.play(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render()
    if truncated or dones:
        obs, _ = env.reset()
