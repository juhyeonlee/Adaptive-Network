
import numpy as np
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from DQN import DQN
from memory import ReplayMemory, Transition


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
        self.batch_size = args['batch_size']
        self.update_target_step = args['update_target_step']

        self.pred_network = DQN(n_state, n_actions, args)
        if torch.cuda.is_available():
            self.pred_network = self.pred_network.cuda()

        self.target_network = copy.deepcopy(self.pred_network)
        self.count = 0

        self.pred_network_params = list(self.pred_network.parameters())

        self.optimizer = optim.RMSprop(params=self.pred_network_params, lr=args['lr'])

        self.memory = ReplayMemory(args)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_action(self, state):
        # decay epsilon value with step count
        epsilon = max(self.epsilon_end, self.epsilon_start - float(self.count) / float(self.epsilon_step))

        rand_action_idx = [np.random.randint(0, self.n_actions) for t in range(len(state))]
        state = state.reshape(-1, self.n_state)
        state = torch.from_numpy(state).float().to(self.device)

        q_value = self.pred_network(state)
        max_action_idx = torch.argmax(q_value, 1).tolist()

        mask = np.random.rand(len(state)) < epsilon
        action_idx = mask * rand_action_idx + (1 - mask) * max_action_idx

        self.count += 1

        return action_idx

    def learn(self):

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat([torch.tensor(s).unsqueeze(0).float() for s in batch.state]).to(self.device)
        next_state_batch = torch.cat([torch.tensor(s).unsqueeze(0).float() for s in batch.next_state]).to(self.device)
        action_batch = torch.cat([torch.tensor(s).unsqueeze(0) for s in batch.action]).to(self.device)
        reward_batch = torch.cat([torch.tensor(s).unsqueeze(0) for s in batch.reward]).to(self.device)

        state_batch = state_batch.reshape([-1, self.n_state])
        next_state_batch = next_state_batch.reshape([-1, self.n_state])
        action_batch = action_batch.unsqueeze(1)
        cur_selected_q = self.pred_network(state_batch).gather(1, action_batch)

        pred_next_max_action = torch.argmax(self.pred_network(next_state_batch), 1)
        pred_next_max_action = pred_next_max_action.unsqueeze(1)
        next_q_value = self.target_network(next_state_batch).gather(1, pred_next_max_action)
        target_q = reward_batch.unsqueeze(1) + self.discount_factor * next_q_value

        loss = F.mse_loss(cur_selected_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.update_target_step == 0:
            self._update_targets()

    def _update_targets(self):
        self.target_network.load_state_dict(self.pred_network.state_dict())
        print('update target')

    def push_memory(self, *args):
        self.memory.push(*args)

    def get_greedy_action(self, state):
        state = state.reshape(-1, self.n_state)
        state = torch.from_numpy(state).float().to(self.device)
        q_value = self.pred_network(state)
        action_idx = torch.argmax(q_value).item()
        return action_idx




