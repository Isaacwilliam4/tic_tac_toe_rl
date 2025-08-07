import argparse
import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple
from tic_tac_rl.mc.black_jack_env import Blackjack


def dfs_black_jack():
    env = Blackjack()
    states = set()

    def dfs(state: Tuple[int, bool, int]):
        if state in states:
            return
        states.add(state)

        for action in [0, 1]:
            # Reset and replay to reach this exact state
            while True:
                s, _ = env.reset()
                while s != state and not env.is_terminal():
                    a = random.choice(env.get_available_actions())
                    s, _, _ = env.step(a)
                if s == state and not env.is_terminal():
                    break

            try:
                s_prime, r, done = env.step(action)
                if not done:
                    dfs(s_prime)
            except ValueError:
                pass

    init_state, _ = env.reset()
    dfs(init_state)
    return states


def monte_carlo_exploring_starts(num_episodes: int = 50000, gamma: float = 1.0):
    env = Blackjack()
    returns = defaultdict(list)
    Q = defaultdict(lambda: [0.0, 0.0])  # Q[state][action]
    policy = {}

    for episode in range(num_episodes):
        while True:
            state, _ = env.reset()
            if not env.is_terminal():
                break
        first_action = random.choice(env.get_available_actions())

        episode_trace: List[Tuple[Tuple[int, bool, int], int, int]] = []
        s = state
        a = first_action

        while True:
            s_prime, r, done = env.step(a)
            episode_trace.append((s, a, r))
            if done:
                break
            s = s_prime
            a = random.choice(env.get_available_actions())

        G = 0
        visited = set()
        for s, a, r in reversed(episode_trace):
            G = gamma * G + r
            if (s, a) not in visited:
                returns[(s, a)].append(G)
                Q[s][a] = sum(returns[(s, a)]) / len(returns[(s, a)])
                visited.add((s, a))

    for s in Q:
        policy[s] = int(Q[s][1] > Q[s][0])

    return policy, Q


def save_policy(npz_path: str, policy: dict, Q: dict):
    policy_arr = {k: v for k, v in policy.items()}
    Q_arr = {k: v for k, v in Q.items()}
    np.savez_compressed(npz_path, policy=policy_arr, Q=Q_arr)
    print(f"Policy and Q values saved to {npz_path}")


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Exploring Starts for Blackjack")
    parser.add_argument("--num_episodes", type=int, default=50000, help="Number of training episodes")
    parser.add_argument("--save_path", type=str, default="blackjack_policy.npz", help="Path to save .npz file")
    args = parser.parse_args()

    policy, Q = monte_carlo_exploring_starts(num_episodes=args.num_episodes)
    save_policy(args.save_path, policy, Q)


if __name__ == "__main__":
    main()
