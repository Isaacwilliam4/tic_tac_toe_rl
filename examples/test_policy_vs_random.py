import numpy as np
import random
from tic_tac_rl.env import TicTacToe
import argparse

parser = argparse.ArgumentParser(description="Evaluate a policy against random play in Tic Tac Toe.")
parser.add_argument('npz_path', type=str, default='policy_results.npz', help='Path to the .npz file containing the policy.')
parser.add_argument('num_games', type=int, default=1000, help='Number of games to play for evaluation.')
parser.add_argument('--debug', action='store_true', help='debug mode')
args = parser.parse_args()


def load_policy(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data['policy'][()], data['value_function'][()]  # both stored as dicts


def random_policy(env):
    return random.choice(env.get_available_actions())


def get_state_key(env):
    """Return the state key including current player, matching training format."""
    return tuple(env.board.reshape(-1)) + (env.current_player,)


def play_game(policy, env, value, debug=False):
    env.reset()
    policy_player = random.choice([1, -1])  # Randomly choose if policy goes first or second
    env.current_player = 1  # Always start with X as per game rules
    if debug:
        print("Game Start:")
        print(f'Policy plays as: {"X" if policy_player == 1 else "O"}\n')

    while True:
        if env.get_current_player() == policy_player:
            # Policy move
            state_key = get_state_key(env)
            action = policy.get(state_key)
            if action not in env.get_available_actions():
                action = random_policy(env)  # Fallback
        else:
            # Random move
            action = random_policy(env)

        env.play(action)
        if debug:
            state_key = get_state_key(env)
            print(f"Value of state: {value.get(state_key, 0.0)}")
            env.print_board()
            print()

        if env.is_terminal():
            winner = env.check_winner()
            if winner is None:
                return None  # Draw
            return winner == policy_player  # True if policy wins, False if loses


def evaluate_policy_vs_random(npz_path, num_games=1000, debug=False):
    policy, value = load_policy(npz_path)
    env = TicTacToe()

    results = {1: 0, -1: 0, 0: 0}  # wins, losses, draws
    for _ in range(num_games):
        result = play_game(policy, env, value, debug)
        if result is None:
            results[0] += 1
        elif result:
            results[1] += 1
        else:
            results[-1] += 1

    print(f"Out of {num_games} games:")
    print(f"Policy wins:  {results[1]}")
    print(f"Random wins:  {results[-1]}")
    print(f"Draws:        {results[0]}")


if __name__ == "__main__":
    evaluate_policy_vs_random(args.npz_path, args.num_games, args.debug)
