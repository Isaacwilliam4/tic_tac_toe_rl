import numpy as np
from tic_tac_rl.dp.tic_tac_toe_env import TicTacToe
import random

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