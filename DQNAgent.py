
import numpy as np
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from DQN import DQN


# TODO: train & load implement
class DQNAgent:
    def __init__(self, n_state, n_actions, action_space, args): # n_state, argsaction_space, discount_factor, agent_num, epsilon, lr):

        self.n_state = n_state
        self.n_actions = n_actions
        self.action_space = action_space
        self.discount_factor = args['gamma']
        self.epsilon_start = args['epsilon_start']
        self.epsilon_end = args['epsilon_end']
        self.epsilon_step = args['epsilon_step']

        self.pred_network = DQN(n_state, n_actions, args)
        if torch.cuda.is_available():
            self.pred_network = self.pred_network.cuda()

        self.target_network = copy.deepcopy(self.pred_network)
        self.count = 0

        self.pred_network_params = list(self.pred_network.parameters())

        self.optimizer = optim.RMSprop(params=self.pred_network_params, lr=args['lr'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_action(self, state):
        # decay epsilon value with step count
        epsilon = max(self.epsilon_end, self.epsilon_start - float(self.count) / float(self.epsilon_step))

        state = state.reshape(-1, self.n_state)
        # state = state / 100.0  # rescale state
        q_value = self.pred_network(state)
        # epsilon = 0.
        if np.random.rand() < epsilon:
            # take random action
            action_idx = np.random.randint(0, self.n_actions)
        else:
            # take greedy action
            action_idx = torch.argmax(q_value).item()

        self.count += 1

        return action_idx

    def learn(self, state, action, reward, next_state, update_step):

        state = np.reshape(state, [-1, self.n_state])
        next_state = np.reshape(next_state, [-1, self.n_state])
        action = torch.tensor([int(action)]).unsqueeze(0).to(self.device)
        cur_selected_q = self.pred_network(state).gather(1, action)

        pred_next_max_action = torch.argmax(self.pred_network(next_state))
        pred_next_max_action = pred_next_max_action.unsqueeze(0).unsqueeze(0)
        next_q_value = self.target_network(next_state).gather(1, pred_next_max_action)
        target_q = reward + self.discount_factor * next_q_value

        loss = F.mse_loss(cur_selected_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % update_step == 0:
            self._update_targets()

    def _update_targets(self):
        self.target_network.load_state_dict(self.pred_network.state_dict())
        print('update target')

    def get_greedy_action(self, state):
        state = state.reshape(-1, self.n_state)
        q_value = self.pred_network(state)
        action_idx = torch.argmax(q_value).item()
        return action_idx




