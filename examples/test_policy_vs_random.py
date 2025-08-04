import numpy as np
import random
from tic_tac_rl.env import TicTacToe
import argparse

args = argparse.ArgumentParser(description="Evaluate a policy against random play in Tic Tac Toe.")
args.add_argument('npz_path', type=str, default='policy_results.npz', help='Path to the .npz file containing the policy.')
args.add_argument('num_games', type=int, default=1000, help='Number of games to play for evaluation.')

def load_policy(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data['policy'][()]  # stored as a dict

def random_policy(env):
    return random.choice(env.get_available_actions())

def play_game(policy, env):
    state, _ = env.reset()
    while True:
        if env.get_current_player() == 1:
            # Trained policy plays
            action = policy.get(state)
            if action not in env.get_available_actions():
                # Fallback to random if illegal (shouldn't happen)
                action = random_policy(env)
        else:
            # Random player
            action = random_policy(env)

        state, reward, done = env.play(action)

        if done:
            winner = env.check_winner()
            return winner  # 1 = policy wins, -1 = random wins, None = draw

def evaluate_policy_vs_random(npz_path, num_games=1000):
    policy = load_policy(npz_path)
    env = TicTacToe()

    results = {1: 0, -1: 0, 0: 0}  # policy wins, random wins, draws
    for _ in range(num_games):
        winner = play_game(policy, env)
        if winner is None:
            results[0] += 1
        else:
            results[winner] += 1

    print(f"Out of {num_games} games:")
    print(f"Policy wins:  {results[1]}")
    print(f"Random wins:  {results[-1]}")
    print(f"Draws:        {results[0]}")

if __name__ == "__main__":
    args = args.parse_args()
    evaluate_policy_vs_random(args.npz_path, args.num_games)
