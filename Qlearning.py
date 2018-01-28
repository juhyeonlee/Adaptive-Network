
import numpy as np
from collections import defaultdict
import random


class QLearningAgent:
    def __init__(self, action_space, ep_length):
        # actions = [1, 2, 3, 4, 5, 6]
        #self.n_actions = n_actions
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.learning_rate = 0.01
        self.discount_factor = 0.7
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epislon_step =100 # 500
        self.q_table = defaultdict(lambda: [0.0] * self.n_actions)
        self.count = 0
        self.ep_length = ep_length

    # update q function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        #TODO: dimension이 next state 에서 달라짐 왜냐면 agent 각각이 소멸되니까? 그러면 각각의 player의 node 번호를 기억했다가 걔네들을 trace 해야되는건가? player수도 가변적임..
        #TODO: multi-arm bandit의 가까운 문제인건가?
        action_idx = self.action_space.index(str(action))
        current_q = self.q_table[state][action_idx]
        # using Bellman Optimality Equation to update q function
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action_idx] += self.learning_rate * (new_q - current_q)

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state, epsilon, steps):
        #epsilon = 0.4 #
        #epsilon =  max(self.epsilon_end, self.epsilon_start - float(self.count) / float(self.epislon_step))
        epsilon = np.sqrt(self.ep_length ** 2 - steps ** 2)
        if np.random.rand() < epsilon:
            # take random action
            action_idx = np.random.choice(self.n_actions)
        else:
            # take action according to the q function table
            state_action = self.q_table[state]
            action_idx = self.arg_max(state_action)
            #print('arg_max result-action_idx:',action_idx) # for checking arg_max function
        action = float(self.action_space[action_idx])
        # print('corresponding action:', action)
        #action = action - 3
        self.count += 1
        return action

    def get_greedy_action(self, state):
        state_action = self.q_table[state]
        action_idx = self.arg_max(state_action)
        #action  = action - 3
        action = self.action_space[action_idx]
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    def print_qtable(self):
        for k, v in self.q_table.items():
            print(k, np.argmax(v))



