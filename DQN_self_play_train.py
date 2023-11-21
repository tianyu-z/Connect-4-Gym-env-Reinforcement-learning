from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import BabyPlayer
from stable_baselines3 import DQN
from connect_four_gymnasium.players.ModelPlayer import ModelPlayer
from connect_four_gymnasium.tools import EloLeaderboard
from net import CustomAZDQNPolicy
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os


class CustomEvalCallback(BaseCallback):
    def __init__(
        self, eval_env, eval_freq, log_path, checkpoint_path, checkpoint_freq, verbose=1
    ):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq
        self.best_elo = -float("inf")
        # Ensure the checkpoint directory exists
        os.makedirs(checkpoint_path, exist_ok=True)
        self.name_prefix = "DQN_4096"

    def _on_step(self) -> bool:
        # Checkpointing
        if self.n_calls % self.checkpoint_freq == 0:
            checkpoint_file = os.path.join(
                self.checkpoint_path, f"{self.name_prefix}_{self.n_calls}"
            )
            self.model.save(checkpoint_file)
        if self.n_calls % self.eval_freq == 0:
            elo = EloLeaderboard().get_elo(opponent, num_matches=250)
            if self.verbose > 0:
                print(f"Step: {self.n_calls}: Your Elo: {elo}")
            if elo > self.best_elo:
                self.best_elo = elo
                self.model.save(self.log_path + "best_model_DQN")
        return True


# Set up the logger, specify the log directory
log_dir = "./logs/"
logger = configure(log_dir, ["stdout", "tensorboard"])
# Set the device for Stable Baselines3
device = "cuda" if torch.cuda.is_available() else "cpu"
env = ConnectFourEnv()
model = DQN(CustomAZDQNPolicy, env, batch_size=8192, verbose=1)  # init training
# model = DQN.load("./eval_logs/best_model_DQN.zip")  # resume training

# Set the environment for the model
opponent = ModelPlayer(model, name="yourself")
env.change_opponent(opponent)
model.set_env(env)

eval_env = ConnectFourEnv()
eval_freq = 1000
checkpoint_freq = 10000
eval_log_path = "./eval_logs/"
checkpoint_path = "./DQNmodels/"
callback = CustomEvalCallback(
    eval_env,
    eval_freq=eval_freq,
    log_path=eval_log_path,
    checkpoint_path=checkpoint_path,
    checkpoint_freq=checkpoint_freq,
)

model.learn(total_timesteps=20000000, callback=callback)
