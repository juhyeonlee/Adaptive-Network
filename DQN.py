import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_state, n_actions, args):
        super(DQN, self).__init__()

        self.inputs_dim = n_state
        self.hidden_dim = args['hidden_dim']

        self.fc1 = nn.Linear(self.inputs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, n_actions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):
        inputs = torch.from_numpy(inputs).float().to(self.device)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        outputs = self.fc3(x)
        return outputs


# hidden_dim = 32

