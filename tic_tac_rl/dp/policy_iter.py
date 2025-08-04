import numpy as np
from typing import Dict, Tuple
from tic_tac_rl.env import TicTacToe
from tic_tac_rl.utils import get_initial_states_policy_vf

def get_player_turn(state):
    return 1 if sum(state) % 2 == 0 else -1


def evaluate_policy(env, V, policy, states, theta=1e-4, gamma=1.0) -> Dict[Tuple[int], float]:
    while True:
        delta = 0
        for state in states:
            env.board = np.array(state).reshape(3, 3)
            env.current_player = get_player_turn(state)
            if env.is_terminal():
                V[state] = 0.0
                continue

            old_v = V[state]
            action = policy.get(state)
            if action is None:
                continue

            env.board[action] = env.current_player
            next_state = env.get_state()
            done = env.is_terminal()
            reward = 0
            if done:
                winner = env.check_winner()
                reward = 1 if winner == env.current_player else -1 if winner == -env.current_player else 0

            new_v = reward + gamma * V.get(next_state, 0.0)
            delta = max(delta, abs(new_v - old_v))
            V[state] = new_v

        if delta < theta:
            break

    return V


def policy_iteration():
    env = TicTacToe()
    states, policy, V = get_initial_states_policy_vf(env)

    def improve_policy():
        policy_stable = True
        for state in policy.keys():
            env.board = np.array(state).reshape(3, 3)
            old_action = policy[state]
            actions = env.get_available_actions()
            best_action_value = float('-inf')
            best_action = None
            for action in actions:
                next_state, reward, done = env.play(action)
                action_value = reward + V.get(next_state, 0.0)
                if action_value > best_action_value:
                    best_action_value = action_value
                    best_action = action
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False
        return not policy_stable

    
    while improve_policy():
        V = evaluate_policy(env, V, policy, states)

    return policy, V

if __name__ == "__main__":
    policy, V = policy_iteration()
    np.savez('policy_results.npz', policy=policy, value_function=V)