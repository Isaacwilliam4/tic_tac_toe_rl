
from .env import TicTacToe
import numpy as np
import random

def get_initial_states_policy_vf(env:TicTacToe):
    visited = set()
    policy = {}
    vf = {}
    def dfs(state, player):
        env.board = np.array(state).reshape(3, 3)
        env.current_player = player
        s = tuple(state)
        if s in visited:
            return
        visited.add(s)
        if env.is_terminal():
            return
        
        actions = env.get_available_actions()
        policy[s] = random.choice(actions) if actions else None
        vf[s] = 0.0
        for action in actions:
            env.board[action] = player
            dfs(env.get_state(), -player)
            env.board[action] = 0
    dfs(env.get_state(), env.current_player)
    return visited, policy, vf