import torch
from tqdm import tqdm

from models.QNet import QNetwork
from agent import World, Agent
from evaluator.BaseEvaluator import BaseEvaluator

from torch.utils.tensorboard import SummaryWriter


class args:
    num_columns = 13
    num_rows = 13
    num_agents = 10

    num_scenarios = 1000

class Evaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__('DQN_DDQN')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net_DQN = QNetwork().to(self.device)
        self.policy_net_DQN.load_model('runs/DQN/run_5/model.pt')
        self.policy_net_DQN.eval()

        self.policy_net_DDQN = QNetwork().to(self.device)
        self.policy_net_DDQN.load_model('runs/DDQN/run_1/model.pt')
        self.policy_net_DDQN.eval()

        self.writer_DQN = SummaryWriter(self.get_eval_directory() + '/DQN')
        self.writer_DDQN = SummaryWriter(self.get_eval_directory() + '/DDQN')

    def evaluate(self):
        world = World(args.num_columns, args.num_rows, args.num_agents)
        world.eval()
        total_reward = 0.0
        progess_bar = tqdm(total=args.num_scenarios, desc="Scenarios completed", leave=True, unit_scale=True, unit='Scenario', colour='green')

        for scenario in range(args.num_scenarios):
            # DQN
            dones = [False for _ in range(world.num_bots)]
            episode_reward = 0.0
            arrived_at_goal = 0

            while not all(dones):
                for agent_idx in range(world.num_bots):
                    if dones[agent_idx]:
                        continue
                    agent: Agent = world.agent_lists[agent_idx]
                    state = agent.get_state()
                    action = agent.get_action(self.policy_net_DQN, state)
                    # print(f"Agent {agent_idx} is moving in epoch {epoch} with action {action}")
                    reward, done = agent.perform_action(action)

                    if done and reward > 0:
                        arrived_at_goal += 1
                    #     print(f"Agent {agent_idx} has reached the goal in epoch {epoch} with reward {reward}!")

                    total_reward += reward
                    episode_reward += reward


                    if done:
                        dones[agent_idx] = True
            
            self.writer_DQN.add_scalar('Arrival rate/scenario', arrived_at_goal/args.num_agents, scenario)
            self.writer_DQN.add_scalar('Reward/scenario', episode_reward, scenario)
            self.writer_DQN.add_scalar('Total reward', total_reward, scenario)
            

            episode_reward = 0.0
            arrived_at_goal = 0

            # soft reset world
            world.reset()
            world.eval()

            # DDQN
            dones = [False for _ in range(world.num_bots)]
            episode_reward = 0.0
            arrived_at_goal = 0

            while not all(dones):
                for agent_idx in range(world.num_bots):
                    if dones[agent_idx]:
                        continue
                    agent: Agent = world.agent_lists[agent_idx]
                    state = agent.get_state()
                    action = agent.get_action(self.policy_net_DDQN, state)
                    # print(f"Agent {agent_idx} is moving in epoch {epoch} with action {action}")
                    reward, done = agent.perform_action(action)

                    if done and reward > 0:
                        arrived_at_goal += 1
                    #     print(f"Agent {agent_idx} has reached the goal in epoch {epoch} with reward {reward}!")

                    total_reward += reward
                    episode_reward += reward


                    if done:
                        dones[agent_idx] = True

            self.writer_DDQN.add_scalar('Arrival rate/scenario', arrived_at_goal/args.num_agents, scenario)
            self.writer_DDQN.add_scalar('Reward/scenario', episode_reward, scenario)
            self.writer_DDQN.add_scalar('Total reeward', total_reward, scenario)

            # hard reset world
            world = World(args.num_columns, args.num_rows, args.num_agents)
            world.eval()

            progess_bar.update(1)


        self.writer_DDQN.close()
        self.writer_DQN.close()

def main():
    trainer = Evaluator()
    trainer.evaluate()

if __name__ == "__main__":
    main()