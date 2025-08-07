import random
from tic_tac_rl.dp.tic_tac_toe_env import TicTacToe

def random_policy(actions):
    return random.choice(actions)

def play_random_game():
    env = TicTacToe()
    state, player = env.reset()
    print("Initial Board")
    print_board(state)

    while True:
        actions = env.get_available_actions()
        action = random_policy(actions)
        state, reward, done = env.play(action)

        print(f"\nPlayer {env.get_current_player() * -1} played {action}")
        print_board(state)

        if done:
            winner = env.check_winner()
            if winner is None:
                print("\nGame ended in a draw.")
            else:
                print(f"\nPlayer {winner} wins!")
            break

def print_board(state):
    board = ["X" if x == 1 else "O" if x == -1 else " " for x in state]
    for i in range(0, 9, 3):
        print("|".join(board[i:i+3]))
        if i < 6:
            print("-+-+-")

if __name__ == "__main__":
    play_random_game()
