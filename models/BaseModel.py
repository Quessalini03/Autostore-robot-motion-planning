import torch

class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def soft_update(self, target_model, local_model, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau), helps stabilize training
        Args:
            target: target network
            source: source network
            tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)