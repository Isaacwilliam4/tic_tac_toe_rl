import numpy as np
from typing import Dict, Tuple
from tic_tac_rl.env import TicTacToe
import argparse
import random

parser = argparse.ArgumentParser(description="Policy Iteration for Tic Tac Toe.")
parser.add_argument('npz_path', type=str, default='policy_results.npz', help='Path to save the policy and value function.')
parser.add_argument('theta', type=float, default=1e-4, help='Convergence threshold for value function updates.')
args = parser.parse_args()

def get_initial_states_policy_vf(env: TicTacToe):
    visited = set()
    policy = {}
    vf = {}

    def dfs(state_board, player):
        env.board = np.array(state_board).reshape(3, 3)
        env.current_player = player
        s = tuple(state_board) + (player,)
        if s in visited:
            return
        visited.add(s)
        if env.is_terminal():
            vf[s] = 0.0
            policy[s] = None
            return

        actions = env.get_available_actions()
        policy[s] = random.choice(actions)
        vf[s] = 0.0
        for action in actions:
            env.board[action] = player
            next_board = env.get_state()
            dfs(next_board, -player)
            env.board[action] = 0

    dfs(env.get_state(), env.current_player)
    return visited, policy, vf

def evaluate_policy(env, V, policy, states, theta=1e-4, gamma=0.99):
    while True:
        delta = 0.0
        for state in states:
            board = np.array(state[:-1]).reshape(3, 3)
            player = state[-1]
            env.board = board
            env.current_player = player

            if env.is_terminal():
                V[state] = 0.0
                continue

            old_v = V[state]
            action = policy.get(state)
            if action is None:
                continue

            env.board[action] = player
            next_board = env.get_state()
            done = env.is_terminal()
            reward = 0
            if done:
                winner = env.check_winner()
                reward = 1 if winner == player else 0
            env.board[action] = 0

            next_key = next_board + (-player,)
            new_v = reward - gamma * V.get(next_key, 0.0)

            V[state] = new_v
            delta = max(delta, abs(new_v - old_v))

        if delta < theta:
            break
    return V

def policy_iteration(theta, gamma=0.99):
    env = TicTacToe()
    states, policy, V = get_initial_states_policy_vf(env)

    def improve_policy():
        policy_stable = True
        for state in list(policy.keys()):
            board = np.array(state[:-1]).reshape(3, 3)
            player = state[-1]
            env.board = board
            env.current_player = player

            if env.is_terminal():
                continue

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
                policy_stable = False
                policy[state] = best_action

        return not policy_stable

    while improve_policy():
        V = evaluate_policy(env, V, policy, states, theta=theta, gamma=gamma)

    return policy, V

if __name__ == "__main__":
    policy, V = policy_iteration(args.theta)
    np.savez(args.npz_path, policy=policy, value_function=V)
