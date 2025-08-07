from tic_tac_rl.mc.black_jack_env import Blackjack


env = Blackjack()
state, _ = env.reset()
done = False

while not done:
    env.print_state()
    action = int(input("Enter 1 for hit, 0 for stick: "))
    state, reward, done = env.step(action)

env.print_state()
print("Final reward:", reward)
