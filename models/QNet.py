import torch
from collections import namedtuple, deque
import random
import numpy as np
import os

from models.BaseModel import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(BaseModel):
    def __init__(self, save_path='saved_models/DQN'):
        super().__init__()
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.conv1 = torch.nn.Conv2d(3, 8, 3)
        self.conv2 = torch.nn.Conv2d(8, 16, 3)
        
        self.fc1 = torch.nn.Linear(18, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 5)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, state):
        observation = state["observation"]
        distance = state["goal"]
        observation = observation.to(device)
        distance = distance.to(device)
        
        observation = torch.nn.functional.relu(self.conv1(observation))
        observation = torch.nn.functional.relu(self.conv2(observation))
        observation = observation.view(-1)

        x = torch.cat([observation, distance], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        y = self.fc3(x)
        return y
    
    @torch.no_grad()
    def predict(self, state):
        return self.forward(state)
    
class QLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.HuberLoss()
    
    def forward(self, output, target):
        return self.loss(output, target)
    
Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ReplayMemory(object):
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, next_state, done)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])

        return (states, torch.tensor(actions, dtype=torch.int64).to(device), torch.tensor(rewards, dtype=torch.float32).to(device),
                next_states, torch.tensor(dones, dtype=torch.bool).to(device))
    
def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    observation = [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]] * 3
    print(observation)
    observation = torch.tensor(observation, dtype=torch.float32).to(device)
    observation = observation.unsqueeze(0)
    distance = [0, 0]
    distance = torch.tensor(distance, dtype=torch.float32).to(device)
    state = {
        "observation": observation,
        "goal": distance
    }

    model = QNetwork().to(device)
    mock_loss = QLoss().to(device)
    mock_target = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    output = model(state)
    loss = mock_loss(output, mock_target)
    optimizer.zero_grad()
    loss.backward()
    print(loss)

if __name__ == "__main__":
    # main()
    source = torch.randint(0, 10, (3, 5))
    print(source)
    max_indexes = source.argmax(dim=1)
    print(max_indexes)
    result = torch.randint(0, 10, (3, 5))
    print(result)
    result = result.gather(1, max_indexes.unsqueeze(1))
    print(result)
