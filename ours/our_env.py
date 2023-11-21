from __future__ import annotations
import numpy as np
import copy
from abc import ABC, abstractmethod
from enum import IntEnum, Enum
import numpy as np
import random


class Outcome(Enum):
    WIN_P1 = 0
    WIN_P2 = 1
    DRAW = 2
    NOT_DONE = 3


class Player(IntEnum):
    ONE = 0
    TWO = 1

    def switch(self) -> Player:
        return Player.ONE if self == Player.TWO else Player.TWO

    def clone(self) -> Player:
        return Player.ONE if self == Player.ONE else Player.TWO


class BaseEnv(ABC):
    # State vars
    board: np.ndarray
    turns: int
    curr_player: Player
    done: bool
    outcome: Outcome | None
    name: str
    BOARD_SHAPE: tuple[int, ...]
    NUM_EXTRA_INFO: int
    ACTION_DIM: int
    MAX_TRAJ_LEN: int  # Number of possible turns + 1

    def __init__(
        self,
        initial_board: np.array | None = None,
        use_conv: bool = True,
        lambd: float = 2.0,
    ) -> None:
        # Get user specified vars
        self.initial_board = initial_board
        self.use_conv = use_conv
        self.lambd = lambd

        # Get env shapes/dims
        self.BOARD_DIM = np.product(self.BOARD_SHAPE).astype(int)
        self.FLAT_STATE_DIM = self.BOARD_DIM + self.NUM_EXTRA_INFO

        self.CONV_SHAPE = (1, *self.BOARD_SHAPE[1:])
        self.CONV_STATE_DIM = (
            self.BOARD_SHAPE[0] + self.NUM_EXTRA_INFO,
            *self.BOARD_SHAPE[1:],
        )

        if self.use_conv:
            self.STATE_DIM = self.CONV_STATE_DIM
        else:
            self.STATE_DIM = self.FLAT_STATE_DIM

    @abstractmethod
    def reset(self) -> np.ndarray:
        ...

    @abstractmethod
    def place_piece(self, action: int) -> None:
        ...

    def evaluate_outcome(self) -> Outcome:
        ...

    def step(self, action: int) -> tuple[np.ndarray, bool]:
        self.place_piece(action)

        self.outcome = self.evaluate_outcome()
        if self.outcome in [Outcome.WIN_P1, Outcome.WIN_P2, Outcome.DRAW]:
            self.done = True
        else:
            self.turns += 1
            self.curr_player = self.curr_player.switch()

        return self.done

    @abstractmethod
    def get_masks(self) -> np.ndarray:
        ...

    def get_curr_player(self) -> Player:
        return self.curr_player

    def get_log_reward(self) -> float:
        assert self.done

        if self.outcome == Outcome.DRAW:
            return (0, 0)

        if self.outcome == Outcome.WIN_P1:
            return (self.lambd, -self.lambd)

        return (-self.lambd, self.lambd)

    @abstractmethod
    def get_extra_info(self) -> np.ndarray:
        ...

    def conv_obs(self) -> np.ndarray:
        extra_info = self.get_extra_info()
        obs = np.concatenate(
            [self.board] + [np.ones(self.CONV_SHAPE) * info for info in extra_info],
            axis=0,
        ).astype(np.float32)

        return obs

    def flat_obs(self) -> np.ndarray:
        obs = np.empty(self.FLAT_STATE_DIM, dtype=np.float32)
        obs[: self.BOARD_DIM] = self.board.flatten()

        extra_info = self.get_extra_info()
        for i, info in enumerate(extra_info):
            obs[self.BOARD_DIM + i] = info

        return obs

    def obs(self) -> np.ndarray:
        return self.conv_obs() if self.use_conv else self.flat_obs()

    @abstractmethod
    def render(self) -> None:
        ...


def create_win_masks(num_rows: int, num_cols: int) -> np.ndarray:
    # Horizontal wins
    horizontal_wins = []
    for row in range(num_rows):
        for col in range(num_cols - 3):
            curr_win = np.zeros((num_rows, num_cols))
            curr_win[row, col : col + 4] = 1
            horizontal_wins.append(curr_win)

    # Vertical wins
    vertical_wins = []
    for row in range(num_rows - 3):
        for col in range(num_cols):
            curr_win = np.zeros((num_rows, num_cols))
            curr_win[row : row + 4, col] = 1
            vertical_wins.append(curr_win)

    # Diagonal wins
    diagonal_wins = []
    for row in range(num_rows - 3):
        for col in range(num_cols - 3):
            # Top left to bottom right
            curr_win = np.zeros((num_rows, num_cols))
            curr_win[np.arange(row, row + 4), np.arange(col, col + 4)] = 1
            diagonal_wins.append(curr_win)

            # Top right to bottom left
            curr_win = np.zeros((num_rows, num_cols))
            curr_win[np.arange(row, row + 4)[::-1], np.arange(col, col + 4)] = 1
            diagonal_wins.append(curr_win)

    wins = horizontal_wins + vertical_wins + diagonal_wins
    wins = np.concatenate([np.expand_dims(win, axis=0) for win in wins], axis=0)
    return wins


class Connect4Env(BaseEnv):
    def __init__(
        self,
        num_rows: int = 6,
        num_cols: int = 7,
        initial_board=None,
        use_conv=True,
        lambd=2,
    ) -> None:
        self.name = "Connect4"
        self.num_rows = 6
        self.num_cols = 7

        self.BOARD_SHAPE = (2, num_rows, num_cols)
        self.NUM_EXTRA_INFO = 2
        self.ACTION_DIM = num_cols
        self.MAX_TRAJ_LEN = num_rows * num_cols + 1
        self.WIN_MASKS = create_win_masks(num_rows, num_cols)

        super().__init__(initial_board, use_conv, lambd)
        self.reset()

    def reset(self) -> None:
        if self.initial_board is not None:
            self.board = self.initial_board.copy()
            self.turns = self.board.sum()
            self.curr_player = Player.ONE if self.turns % 2 == 0 else Player.TWO
        else:
            self.board = np.zeros(self.BOARD_SHAPE, dtype=np.float32)
            self.turns = 0
            self.curr_player = Player.ONE

        self.done = False
        self.outcome = Outcome.NOT_DONE

    def get_extra_info(self):
        return [
            int(self.curr_player),
            int(self.turns),
        ]

    def place_piece(self, action: int) -> None:
        num_pieces_column = int(self.board.sum(axis=(0, 1))[action])
        row = (
            self.num_rows - num_pieces_column - 1
        )  # E.g. if there are 6 rows and 2 pieces already, place at row with index 3
        col = action
        assert ~np.any(
            self.board[:, row, col]
        ), f"Cannot override a placed piece at ({row}, {col})"

        # Update the internal state
        self.board[self.curr_player, row, col] = 1

    def won(self, player: Player):
        x = (self.WIN_MASKS * self.board[player]).sum(axis=(1, 2))
        x = x.max()
        return x == 4

    def evaluate_outcome(self):
        if self.won(Player.ONE):
            return Outcome.WIN_P1
        elif self.won(Player.TWO):
            return Outcome.WIN_P2
        elif self.turns + 1 >= self.num_rows * self.num_cols:
            return Outcome.DRAW
        else:
            return Outcome.NOT_DONE

    def get_masks(self) -> np.ndarray:
        return self.board.sum(axis=(0, 1)) < self.num_rows

    def render(self) -> None:
        board = self.board[0] - self.board[1]

        print()

        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row, col] == 1:
                    print("X", end=" ")
                elif board[row, col] == -1:
                    print("O", end=" ")
                else:
                    print("-", end=" ")
            print()

        # Print column numbers
        for col in range(board.shape[1]):
            print(col, end=" ")

        print()
        print()


def get_env_dims(env):
    """
    Get the state dimensions and action dimensions for a given environment.

    Args:
    - env: The game environment.

    Returns:
    - state_dims: Dimensions of the state (observation).
    - action_dims: Number of possible actions.
    """
    # Get the state dimensions from the observation space shape
    state_dims = env.obs().shape

    # Get the action dimensions from the action space
    action_dims = len(env.get_masks())

    return state_dims, action_dims


def test():
    env = Connect4Env()

    for i in range(25):
        if not env.done:
            print(env.obs().shape)
            env.step(i % 7)
            env.render()

    print(env.outcome)


if __name__ == "__main__":
    test()
