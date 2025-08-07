import numpy as np
from typing import List, Tuple, Optional


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O

    def reset(self) -> Tuple[Tuple[int], int]:
        self.board.fill(0)
        self.current_player = 1
        return self.get_state(), self.current_player

    def get_state(self) -> Tuple[int]:
        return tuple(self.board.reshape(-1))

    def get_current_player(self) -> int:
        return self.current_player

    def get_available_actions(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def is_terminal(self) -> bool:
        return self.check_winner() is not None or not (self.board == 0).any()

    def check_winner(self) -> Optional[int]:
        for i in range(3):
            row_sum = self.board[i].sum()
            col_sum = self.board[:, i].sum()
            if abs(row_sum) == 3:
                return int(np.sign(row_sum))
            if abs(col_sum) == 3:
                return int(np.sign(col_sum))

        diag1_sum = self.board.trace()
        diag2_sum = np.fliplr(self.board).trace()
        if abs(diag1_sum) == 3:
            return int(np.sign(diag1_sum))
        if abs(diag2_sum) == 3:
            return int(np.sign(diag2_sum))

        return None

    def play(self, action: Tuple[int, int]) -> Tuple[Tuple[int], int, bool]:
        i, j = action
        if self.board[i, j] != 0:
            raise ValueError("Invalid action: Cell already taken.")

        self.board[i, j] = self.current_player
        winner = self.check_winner()

        if winner == self.current_player:
            reward = 1
        elif winner == -self.current_player:
            reward = -1
        elif not (self.board == 0).any():
            reward = 0  # draw
        else:
            reward = 0

        done = self.is_terminal()
        self.current_player *= -1
        return self.get_state(), reward, done
    
    def print_board(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for row in self.board:
            print(" | ".join(symbols[x] for x in row))
            print("-" * 9)
        print()