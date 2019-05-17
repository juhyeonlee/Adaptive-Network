import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, args):
        super(DQN, self).__init__()

        self.inputs_dim = args.n_state
        self.hidden_dim = args.hidden_dim

        self.fc1 = nn.Linear(self.inputs_dim, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        outputs = self.fc2(h)
        return  outputs, h


# hidden_dim = 32

