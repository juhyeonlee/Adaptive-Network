
import random

from env import Environment
from Qlearning import QLearningAgent

if __name__ == '__main__':

    # init parameters
    one_dim = 5
    mu = 4 / 5
    init_txr = 3
    # random_seed = 1 #??

    env = Environment(one_dim, mu, init_txr)

    n_actions = env.n_actions
    agent = QLearningAgent(n_actions)

    ep_length = 1
    num_ep = 1

    for episode in range(num_ep):
        beta = random.uniform(0.0, 0.3)  #TODO: specified particular distributions for link failure rate
        state = env.reset(beta, init_txr)

        for steps in range(ep_length):
            action = agent.get_action(state)
            real_action = action - 3 # [0,6] --> [-3, 3]
            next_state, reward = env.step(real_action)

            agent.learn(state, action, reward, next_state)

            state = next_state

            agent.print_qtable()

