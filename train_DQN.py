import torch
from trainer.BaseTrainer import BaseTrainer
from models.QNet import QNetwork, QLoss, ReplayMemory, Experience
from agent import World, Agent
import yaml

from torch.utils.tensorboard import SummaryWriter

class args:
    epsilon = 0.9
    gamma = 0.7
    epsilon_decrement = 0.00001
    num_columns = 13
    num_rows = 13
    state_dimension = 65
    num_agents = 10

    num_epochs = 10000
    learning_rate = 0.0001
    sync_rate = 5
    replay_size = 1000
    batch_size = 128
    time_to_live = 150
    tau = 0.01

class Trainer(BaseTrainer):
    def __init__(self) -> None:
        super().__init__('DQN')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork().to(self.device)
        self.policy_net.train()
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.memory = ReplayMemory(args.replay_size)
        self.criteria = QLoss()
        # default `log_dir` is "runs" - we'll be more specific here
        self.writer = SummaryWriter(self.get_run_directory())
        self.current_loss = float('inf')
        self.model_path = self.get_model_path()

        self.save_hyperparameters(args)

    def train_one(self, state, action, reward, next_state, done):
        output = self.policy_net(state)
        target = output.clone()
        Q_new = reward
        if not done:
            with torch.no_grad():
                Q_new = reward + args.gamma * torch.max(self.target_net(next_state))

        target[action] = Q_new
        loss = self.criteria(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train_replay(self, epoch):
        if len(self.memory) < args.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(args.batch_size)

        preds = torch.stack([self.policy_net(state) for state in states])
        targets = preds.clone()

        for i in range(len(dones)):
            Q_new = rewards[i]
            if not dones[i]:
                with torch.no_grad():
                    Q_new = rewards[i] + args.gamma * torch.max(self.target_net(next_states[i]))

            targets[i][actions[i]] = Q_new
        
        loss = self.criteria(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if loss < self.current_loss:
            self.current_loss = loss
            self.policy_net.save_model(self.model_path)

        self.writer.add_scalar('Loss/train', loss, epoch)
        

    def train(self):
        world = World(args.num_columns, args.num_rows, args.num_agents)

        total_reward = 0.0

        for epoch in range(args.num_epochs):
            dones = [False for _ in range(world.num_bots)]
            episode_reward = 0.0
            arrived_at_goal = 0

            while not all(dones):
                for agent_idx in range(world.num_bots):
                    if dones[agent_idx]:
                        continue
                    agent: Agent = world.agent_lists[agent_idx]
                    state = agent.get_state()
                    with torch.no_grad():
                        action = agent.get_action(self.policy_net, state)
                    # print(f"Agent {agent_idx} is moving in epoch {epoch} with action {action}")
                    reward, done, arrived = agent.perform_action(action)
                    next_state = agent.get_state()
                    # print(next_state)
                    exp = Experience(state, action, reward, next_state, done)

                    if arrived:
                        arrived_at_goal += 1
                    #     print(f"Agent {agent_idx} has reached the goal in epoch {epoch} with reward {reward}!")

                    total_reward += reward
                    episode_reward += reward

                    self.memory.append(exp)
                    self.train_one(state, action, reward, next_state, done)

                    if done:
                        dones[agent_idx] = True
            
            self.writer.add_scalar('Arrival rate/epoch', arrived_at_goal/args.num_agents, epoch)
            self.writer.add_scalar('Reward/episode', episode_reward, epoch)
            self.writer.add_scalar('Reward/epoch', total_reward, epoch)
            
            self.train_replay(epoch)
            self.policy_net.soft_update(self.target_net, self.policy_net, args.tau)

            episode_reward = 0.0
            arrived_at_goal = 0

            if epoch % 1000 == 0:
                print(f"Epoch {epoch} done. Total reward: {total_reward}")

            # reset world
            world = World(args.num_columns, args.num_rows, args.num_agents)

def main():
    trainer = Trainer()
    trainer.train()

if __name__ == "__main__":
    main()