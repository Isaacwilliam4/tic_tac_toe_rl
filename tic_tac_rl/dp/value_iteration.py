import numpy as np
from tic_tac_rl.dp.tic_tac_toe_env import TicTacToe
import argparse
from tqdm import tqdm
from .utils import get_initial_states_policy_vf

parser = argparse.ArgumentParser(description="Policy Iteration for Tic Tac Toe.")
parser.add_argument('npz_path', type=str, default='policy_results.npz', help='Path to save the policy and value function.')
parser.add_argument('theta', type=float, default=1e-4, help='Convergence threshold for value function updates.')
args = parser.parse_args()

def value_iteration(env: TicTacToe, V, policy, states, theta=1e-4, gamma=0.99):
    while True:
        delta = 0.0
        pbar = tqdm(states)
        for state in pbar:
            board = np.array(state[:-1]).reshape(3, 3)
            player = state[-1]
            env.board = board
            env.current_player = player

            if env.is_terminal():
                V[state] = 0.0
                continue

            old_v = V[state]

            old_action = policy[state]
            best_action = None
            best_value = -float("inf")

            for action in env.get_available_actions():
                env.board[action] = player
                next_board = env.get_state()
                done = env.is_terminal()
                reward = 0
                if done:
                    winner = env.check_winner()
                    reward = 1 if winner == player else 0
                env.board[action] = 0

                next_key = next_board + (-player,)
                value = reward - gamma * V.get(next_key, 0.0)

                if value > best_value:
                    best_value = value
                    best_action = action

            if best_action != old_action:
                policy[state] = best_action

            V[state] = best_value
            delta = max(delta, abs(best_value - old_v))
            pbar.set_postfix_str(f"delta={delta}/{theta}")

        if delta < theta:
            break

    return policy, V

def get_policy(theta, gamma=0.99):
    env = TicTacToe()
    states, policy, V = get_initial_states_policy_vf(env)
    return value_iteration(env, V, policy, states, theta, gamma)

if __name__ == "__main__":
    policy, V = get_policy(args.theta)
    np.savez(args.npz_path, policy=policy, value_function=V)
