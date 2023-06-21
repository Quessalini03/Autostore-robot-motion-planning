from typing import List, Tuple

import pytorch_lightning as pl
from config import args
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from agent import World, Agent
from buffer import ReplayBuffer
from dataset import RLDataset
from net import DQN

class DQNLightning(pl.LightningModule):
    """Basic DQN Model."""
    def __init__(
        self,
        batch_size: int = args.batch_size,
        lr: float = args.learning_rate,
        gamma: float = args.gamma,
        sync_rate: int = args.sync_rate,
        replay_size: int = args.replay_size,
        patient_factor: int = args.patient_factor,
        num_columns: int = args.num_columns,
        num_rows: int = args.num_rows,
        num_agents: int = args.num_agents,
        warmup_steps: int = args.warmup_steps
    ) -> None:
        super().__init__()
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.num_agents = num_agents
        self.save_hyperparameters()

        self.net = DQN(args.state_dimension, args.num_actions, 256)
        self.target_net = DQN(args.state_dimension, args.num_actions, 256)
        self.target_net.load_state_dict(self.net.state_dict())
        self.memory = ReplayBuffer(100000)
        self.world = World(num_columns, num_rows, num_agents)

        self.total_reward = 0.0
        self.episode_reward = 0.0

        self.populate(warmup_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        return output
    
    def configure_optimizers(self) -> List[optim.Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def dqn_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        states, actions, rewards, dones, next_states = batch

        preds = self.target_net(states)
        targets = preds.clone()

        for i in range(len(dones)):
            Q_new = rewards[i]
            if not dones[i]:
                Q_new = Q_new + self.hparams.gamma * torch.max(self.target_net(next_states[i]))

            targets[i][actions[i]] = Q_new

        return nn.functional.huber_loss(preds, targets)
    

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        # step through environment with agent
        rewards = []
        dones = []
        for agent_idx in range(self.num_agents):
            reward, done = self.world.agent_lists[agent_idx].play_step(self.net, self.memory)
            rewards.append(reward)
            dones.append(done)

        step_reward = sum(rewards)
        self.episode_reward += step_reward

        self.log("reward", step_reward)

        # calculates training loss
        loss = self.dqn_loss(batch)

        if all(dones):
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self.world = World(self.num_columns, self.num_rows, self.num_agents)

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "reward": step_reward,
                "train_loss": loss,
            }
        )
        self.log("total_reward", self.total_reward, prog_bar=True)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.memory, self.hparams.batch_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def populate(self, steps: int = 1000) -> None:
        for _ in range(steps):
            for agent_idx in range(self.num_agents):
                self.world.agent_lists[agent_idx].play_step(self.net, self.memory)   
