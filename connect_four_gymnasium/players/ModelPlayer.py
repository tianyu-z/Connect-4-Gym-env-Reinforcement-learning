from .Player import Player
import numpy as np


class ModelPlayer(Player):
    def __init__(self, model, name="Model", deteministic=True):
        self.model = model
        self.name = name
        self.deteministic = deteministic

    def play(
        self, observations
    ):  # todo other bot : make able to takes multiples observations
        resample = True
        while resample:
            actions, _states = self.model.predict(
                observations, deterministic=self.deteministic
            )
            resample = self.is_column_occupied(observations, actions)
        return actions

    def is_column_occupied(self, array, col):
        return np.all(array[:, col] != 0)

    def getName(self):
        return self.name

    def isDeterministic(self):
        return self.deteministic

    def getElo(self):
        return None
