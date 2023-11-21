import torch
import copy
import random
import math

from ours.our_env import BaseEnv, Player, Outcome
from tqdm import tqdm

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Agent:
    def sample_actions(self, env: BaseEnv, side: Player):
        pass


class UniformAgent:
    def sample_actions(self, env: BaseEnv, side: Player):
        masks = torch.from_numpy(env.get_masks()).float().to(DEVICE)
        action = masks.multinomial(1).item()
        return action


class HumanAgent:
    def sample_actions(self, env: BaseEnv, side: Player):
        available_actions = env.get_masks().nonzero()[0].tolist()
        env.render()
        action = input("Enter action: ")
        while int(action) not in available_actions:
            action = input(f"Invalid action: {action}. Enter action: ")
        return int(action)


class HeuristicAgent:
    def __init__(self, name: str, depth: int) -> None:
        self.name = name
        self.depth = depth

    def heuristic(self, env: BaseEnv):
        pass

    def get_legal_actions(self, env: BaseEnv):
        return env.get_masks().nonzero()[0]

    def sample_actions(self, env: BaseEnv, side: Player):
        def AB_recurse(env: BaseEnv, curr_depth: int, alpha, beta) -> float:
            if env.done or curr_depth == self.depth:
                heuristic_val = self.heuristic(env)
                if side == Player.TWO:
                    heuristic_val *= -1
                return None, heuristic_val

            best_actions = []

            actions = self.get_legal_actions(env)
            if env.get_curr_player() == side:
                val = float("-inf")
                for action in actions:
                    copy_env = copy.deepcopy(env)
                    copy_env.step(action)
                    _, curr_val = AB_recurse(copy_env, curr_depth + 1, alpha, beta)

                    # if curr_val == val:
                    #     best_actions.append(action)

                    if curr_val > val:
                        val = curr_val
                        best_actions = [action]

                    val = max(val, curr_val)
                    alpha = max(alpha, val)

                    if val >= beta:
                        break

                return best_actions, val
            else:
                val = float("inf")
                for action in actions:
                    copy_env = copy.deepcopy(env)
                    copy_env.step(action)
                    _, curr_val = AB_recurse(copy_env, curr_depth + 1, alpha, beta)
                    val = min(val, curr_val)
                    beta = min(beta, val)

                    if val <= alpha:
                        break
                return None, val

        best_actions, best_val = AB_recurse(env, 0, float("-inf"), float("inf"))
        # print(best_actions, best_val)
        return random.choice(best_actions)


class DecentAgent(HeuristicAgent):
    def heuristic(self, env: BaseEnv):
        masked_board = env.WIN_MASKS * env.board[0], env.WIN_MASKS * env.board[1]

        def get_win(side):
            return float(masked_board[side].sum(axis=(1, 2)).max() == 4)

        def get_num_aligned(side):
            x = (env.WIN_MASKS * env.board[side]).sum(axis=(1, 2))
            x -= 10 * (env.WIN_MASKS * env.board[1 - side]).sum(axis=(1, 2))
            return x.max()

        return (
            0.1 * get_num_aligned(0)
            - 0.1 * get_num_aligned(1)
            + get_win(0)
            - get_win(1)
        )


def play_game(env: BaseEnv, agent_1: Agent, agent_2: Agent, verbose=False):
    while not env.done:
        copy_env = copy.deepcopy(env)
        if env.get_curr_player() == Player.ONE:
            action = agent_1.sample_actions(copy_env, Player.ONE)
        else:
            action = agent_2.sample_actions(copy_env, Player.TWO)
        env.step(action)

    outcome = env.outcome
    if verbose:
        print(f"Outcome: {outcome}")

    env.reset()

    return outcome


def get_matchup_stats(env: BaseEnv, agent_1: torch.nn.Module, agent_2: Agent) -> dict:
    test_res = {
        "WINS": 0,
        "DRAWS": 0,
        "LOSSES": 0,
    }
    num_games = 100

    for side in [Player.ONE, Player.TWO]:
        for _ in tqdm(range(num_games), leave=False):
            if side == Player.ONE:
                outcome = play_game(env, agent_1, agent_2)
            else:
                outcome = play_game(env, agent_2, agent_1)

            if outcome.value == side:
                test_res["WINS"] += 1
            elif outcome == Outcome.DRAW:
                test_res["DRAWS"] += 1
            else:
                test_res["LOSSES"] += 1

    ratios = {f"{name} ratio": val / (num_games * 2) for name, val in test_res.items()}
    return ratios


def test_afn(env: BaseEnv, afn: torch.nn.Module, opp_agent: Agent):
    afn.eval()
    ratios = get_matchup_stats(env, afn, opp_agent)
    afn.train()

    return ratios


def afn_accuracy(afn: torch.nn.Module):
    BATCH_SIZE = 1024

    res = {"Optimal": 0, "Inaccuracy": 0, "Blunder": 0, "Losing move": 0}

    vals = torch.load("state_vals.pt")
    for player in [0, 1]:
        curr_vals = vals[player]
        for batch in range(math.ceil(len(curr_vals) / BATCH_SIZE)):
            start, end = batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE
            state_batch = torch.stack(
                [torch.from_numpy(x[1]).cuda() for x in curr_vals[start:end]], dim=0
            )
            vals_batch = torch.stack([x[2].cuda() for x in curr_vals[start:end]], dim=0)

            optimal_move, optimal_val = (
                vals_batch.max(dim=1).indices,
                vals_batch.max(dim=1).values,
            )

            _, policy = afn(state_batch, player)
            selected_move = policy.argmax(dim=1)
            selected_val = vals_batch.gather(1, selected_move.unsqueeze(1)).squeeze(1)

            is_same_sign = selected_val.sign() == optimal_val.sign()

            # Optimal
            is_optimal = selected_val == optimal_val
            res["Optimal"] += (selected_val == optimal_val).sum()

            # Inaccuracy
            is_small_mistake = (selected_val - optimal_val).abs() <= 1
            is_inaccuracy = (is_same_sign & is_small_mistake) & (~is_optimal)
            res["Inaccuracy"] += is_inaccuracy.sum()

            # Blunder
            is_big_mistake = (selected_val - optimal_val).abs() > 1
            is_blunder = is_same_sign & is_big_mistake
            res["Blunder"] += is_blunder.sum()

            # Losing move
            res["Losing move"] += (~is_same_sign).sum()

    for k, v in res.items():
        res[k] = v / (len(vals[0]) + len(vals[1]))
    return res
