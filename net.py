import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        self.net = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.action_size),
        )

    def forward(self, state):
        return self.net(state.float())