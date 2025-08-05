import numpy as np
from tic_tac_rl.env import TicTacToe
import argparse

parser = argparse.ArgumentParser(description="Play Tic Tac Toe against a trained policy.")
parser.add_argument('npz_path', type=str, default='policy.npz', help='Path to the .npz file containing the policy.')
args = parser.parse_args()


def load_policy(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data['policy'][()]  # dict stored in npz


def print_board(state):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    board = [symbols[x] for x in state]
    for i in range(0, 9, 3):
        print(" | ".join(board[i:i+3]))
        if i < 6:
            print("--+---+--")


def get_human_action(env):
    actions = env.get_available_actions()
    while True:
        try:
            pos = input("Enter your move as row,col (0 to 2): ")
            i, j = map(int, pos.strip().split(","))
            if (i, j) in actions:
                return (i, j)
            else:
                print("Invalid move. Try again.")
        except Exception:
            print("Invalid format. Try again (example: 1,2).")


def state_key(env):
    # key format matches training: (flattened_board_tuple, player_to_move)
    return tuple(env.board.reshape(-1)) + (env.get_current_player(),)


def play_vs_policy(policy, human_player=1):
    env = TicTacToe()
    state, _ = env.reset()

    print("You are", "X" if human_player == 1 else "O")
    print_board(state)

    while True:
        current = env.get_current_player()
        if current == human_player:
            action = get_human_action(env)
        else:
            key = state_key(env)
            action = policy.get(key)
            if action not in env.get_available_actions():
                action = env.get_available_actions()[0]
            print(f"Policy plays: {action[0]+1},{action[1]+1}")

        state, reward, done = env.play(action)
        print_board(state)
        print()

        if done:
            winner = env.check_winner()
            if winner == human_player:
                print("You win!")
            elif winner == -human_player:
                print("Policy wins!")
            else:
                print("It is a draw.")
            break


if __name__ == "__main__":
    policy = load_policy(args.npz_path)
    first = input("Do you want to go first? (y/n): ").strip().lower()
    human = 1 if first == "y" else -1
    play_vs_policy(policy, human_player=human)
