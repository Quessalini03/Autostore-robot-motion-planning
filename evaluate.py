import torch
from tqdm import tqdm
import time
import pandas as pd

from models.QNet import QNetwork
from agent import World, Agent
from evaluator.BaseEvaluator import BaseEvaluator

from torch.utils.tensorboard import SummaryWriter


class args:
    num_columns = 18
    num_rows = 18
    num_agents = 32

    num_scenarios = 1000

class Evaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__('DQN_DDQN_HybridDQN_HybridDDQN')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net_DQN = QNetwork().to(self.device)
        self.policy_net_DQN.load_model('runs/DQN/run_5/model.pt')
        self.policy_net_DQN.eval()

        self.policy_net_DDQN = QNetwork().to(self.device)
        self.policy_net_DDQN.load_model('runs/DDQN/run_1/model.pt')
        self.policy_net_DDQN.eval()

        self.writer_DQN = SummaryWriter(self.get_eval_directory() + '/DQN')
        self.writer_DDQN = SummaryWriter(self.get_eval_directory() + '/DDQN')
        self.writer_HybridDQN = SummaryWriter(self.get_eval_directory() + '/Hybrid_DQN')
        self.writer_HybridDDQN = SummaryWriter(self.get_eval_directory() + '/Hybrid_DDQN')

        self.csv_path = self.get_eval_directory() + '/result.csv'

        self.save_evaluate_params(args)

    def evaluate(self):
        world = World(args.num_columns, args.num_rows, args.num_agents)
        world.eval()
        progess_bar = tqdm(total=args.num_scenarios, desc="Scenarios completed", leave=True, unit_scale=True, unit='Scenario', colour='green')


        total_reward_DQN = 0
        total_reward_DDQN = 0
        total_reward_HybridDQN = 0
        total_reward_HybridDDQN = 0

        arrived_DQN = 0
        arrived_DDQN = 0
        arrived_HybridDQN = 0
        arrived_HybridDDQN = 0

        starting_distance_DQN = 0
        starting_distance_DDQN = 0
        starting_distance_HybridDQN = 0
        starting_distance_HybridDDQN = 0

        travelled_distance_DQN = 0
        travelled_distance_DDQN = 0
        travelled_distance_HybridDQN = 0
        travelled_distance_HybridDDQN = 0

        direction_changes_DQN = 0
        direction_changes_DDQN = 0
        direction_changes_HybridDQN = 0
        direction_changes_HybridDDQN = 0

        total_arrive_time_DQN = 0
        total_arrive_time_DDQN = 0
        total_arrive_time_HybridDQN = 0
        total_arrive_time_HybridDDQN = 0

        for scenario in range(args.num_scenarios):
            # DQN
            dones = [False for _ in range(world.num_bots)]
            episode_reward = 0.0
            arrived_at_goal = 0
            start_time = time.time()

            while not all(dones):
                for agent_idx in range(world.num_bots):
                    if dones[agent_idx]:
                        continue
                    agent: Agent = world.agent_lists[agent_idx]
                    state = agent.get_state()
                    action = agent.get_action(self.policy_net_DQN, state)
                    # print(f"Agent {agent_idx} is moving in epoch {epoch} with action {action}")
                    reward, done, arrived = agent.perform_action(action)

                    if arrived:
                        arrived_at_goal += 1
                        arrived_DQN += 1
                        starting_distance_DQN += agent.starting_distance_to_goal
                        travelled_distance_DQN += agent.distance_travelled
                        direction_changes_DQN += agent.num_direction_changes
                        total_arrive_time_DQN += time.time() - start_time
                    #     print(f"Agent {agent_idx} has reached the goal in epoch {epoch} with reward {reward}!")

                    total_reward_DQN += reward
                    episode_reward += reward


                    if done:
                        dones[agent_idx] = True
            
            self.writer_DQN.add_scalar('Arrival rate/scenario', arrived_at_goal/args.num_agents, scenario)
            self.writer_DQN.add_scalar('Reward/scenario', episode_reward, scenario)
            self.writer_DQN.add_scalar('Total reward', total_reward_DQN, scenario)
            

            episode_reward = 0.0
            arrived_at_goal = 0

            # soft reset world
            world.reset()
            world.eval()

            # DDQN
            dones = [False for _ in range(world.num_bots)]
            episode_reward = 0.0
            arrived_at_goal = 0
            start_time = time.time()

            while not all(dones):
                for agent_idx in range(world.num_bots):
                    if dones[agent_idx]:
                        continue
                    agent: Agent = world.agent_lists[agent_idx]
                    state = agent.get_state()
                    action = agent.get_action(self.policy_net_DDQN, state)
                    # print(f"Agent {agent_idx} is moving in epoch {epoch} with action {action}")
                    reward, done, arrived = agent.perform_action(action)

                    if arrived:
                        arrived_at_goal += 1
                        arrived_DDQN += 1
                        starting_distance_DDQN += agent.starting_distance_to_goal
                        travelled_distance_DDQN += agent.distance_travelled
                        direction_changes_DDQN += agent.num_direction_changes
                        total_arrive_time_DDQN += time.time() - start_time
                    #     print(f"Agent {agent_idx} has reached the goal in epoch {epoch} with reward {reward}!")

                    total_reward_DDQN += reward
                    episode_reward += reward


                    if done:
                        dones[agent_idx] = True

            self.writer_DDQN.add_scalar('Arrival rate/scenario', arrived_at_goal/args.num_agents, scenario)
            self.writer_DDQN.add_scalar('Reward/scenario', episode_reward, scenario)
            self.writer_DDQN.add_scalar('Total reward', total_reward_DDQN, scenario)

            episode_reward = 0.0
            arrived_at_goal = 0

            # soft reset world
            world.reset()
            world.eval()

            # Hybrid DQN
            dones = [False for _ in range(world.num_bots)]
            episode_reward = 0.0
            arrived_at_goal = 0
            start_time = time.time()

            while not all(dones):
                for agent_idx in range(world.num_bots):
                    if dones[agent_idx]:
                        continue
                    agent: Agent = world.agent_lists[agent_idx]
                    state = agent.get_state()
                    # if agent.should_force_policy():
                    #     action = agent.get_action(self.policy_net_DQN, state)
                    #     reward, done, arrived = agent.perform_action(action)
                    #     agent.post_force_policy()
                    if agent.should_move_heuristic(state):
                        action = agent.get_heuristic_action()
                        reward, done, arrived = agent.perform_action(action)
                    else:
                        action = agent.get_action(self.policy_net_DQN, state)
                        reward, done, arrived = agent.perform_action(action)
                        agent.post_normal_policy()
                    # print(f"Agent {agent_idx} is moving in epoch {epoch} with action {action}")

                    if arrived:
                        arrived_at_goal += 1
                        arrived_HybridDQN += 1
                        starting_distance_HybridDQN += agent.starting_distance_to_goal
                        travelled_distance_HybridDQN += agent.distance_travelled
                        direction_changes_HybridDQN += agent.num_direction_changes
                        total_arrive_time_HybridDQN += time.time() - start_time
                    #     print(f"Agent {agent_idx} has reached the goal in epoch {epoch} with reward {reward}!")

                    total_reward_HybridDQN += reward
                    episode_reward += reward


                    if done:
                        dones[agent_idx] = True

            self.writer_HybridDQN.add_scalar('Arrival rate/scenario', arrived_at_goal/args.num_agents, scenario)
            self.writer_HybridDQN.add_scalar('Reward/scenario', episode_reward, scenario)
            self.writer_HybridDQN.add_scalar('Total reward', total_reward_HybridDQN, scenario)

            # soft reset world
            world.reset()
            world.eval()

            # Hybrid DDQN
            dones = [False for _ in range(world.num_bots)]
            episode_reward = 0.0
            arrived_at_goal = 0
            start_time = time.time()

            while not all(dones):
                for agent_idx in range(world.num_bots):
                    if dones[agent_idx]:
                        continue
                    agent: Agent = world.agent_lists[agent_idx]
                    state = agent.get_state()
                    state = agent.get_state()
                    # if agent.should_force_policy():
                    #     action = agent.get_action(self.policy_net_DDQN, state)
                    #     reward, done, arrived = agent.perform_action(action)
                    #     agent.post_force_policy()
                    if agent.should_move_heuristic(state):
                        action = agent.get_heuristic_action()
                        reward, done, arrived = agent.perform_action(action)
                    else:
                        action = agent.get_action(self.policy_net_DDQN, state)
                        reward, done, arrived = agent.perform_action(action)
                        agent.post_normal_policy()

                    if arrived:
                        arrived_at_goal += 1
                        arrived_HybridDDQN += 1
                        starting_distance_HybridDDQN += agent.starting_distance_to_goal
                        travelled_distance_HybridDDQN += agent.distance_travelled
                        direction_changes_HybridDDQN += agent.num_direction_changes
                        total_arrive_time_HybridDDQN += time.time() - start_time
                    #     print(f"Agent {agent_idx} has reached the goal in epoch {epoch} with reward {reward}!")

                    total_reward_HybridDDQN += reward
                    episode_reward += reward


                    if done:
                        dones[agent_idx] = True

            self.writer_HybridDDQN.add_scalar('Arrival rate/scenario', arrived_at_goal/args.num_agents, scenario)
            self.writer_HybridDDQN.add_scalar('Reward/scenario', episode_reward, scenario)
            self.writer_HybridDDQN.add_scalar('Total reward', total_reward_HybridDDQN, scenario)

            # hard reset world
            world = World(args.num_columns, args.num_rows, args.num_agents)
            world.eval()

            progess_bar.update(1)


        self.writer_DQN.add_scalar('Total reward/scenario', total_reward_DQN/args.num_scenarios, 0)
        self.writer_DDQN.add_scalar('Total reward/scenario', total_reward_DDQN/args.num_scenarios, 0)
        self.writer_HybridDQN.add_scalar('Total reward/scenario', total_reward_HybridDQN/args.num_scenarios, 0)
        self.writer_HybridDDQN.add_scalar('Total reward/scenario', total_reward_HybridDDQN/args.num_scenarios, 0)

        self.writer_DQN.add_scalar('Arrival rate/total', arrived_DQN/(args.num_scenarios * args.num_agents), 0)
        self.writer_DDQN.add_scalar('Arrival rate/total', arrived_DDQN/(args.num_scenarios * args.num_agents), 0)
        self.writer_HybridDQN.add_scalar('Arrival rate/total', arrived_HybridDQN/(args.num_scenarios * args.num_agents), 0)
        self.writer_HybridDDQN.add_scalar('Arrival rate/total', arrived_HybridDDQN/(args.num_scenarios * args.num_agents), 0)

        self.writer_DQN.add_scalar('Distance travelled/starting distance', travelled_distance_DQN/starting_distance_DQN, 0)
        self.writer_DDQN.add_scalar('Distance travelled/starting distance', travelled_distance_DDQN/starting_distance_DDQN, 0)
        self.writer_HybridDQN.add_scalar('Distance travelled/starting distance', travelled_distance_HybridDQN/starting_distance_HybridDQN, 0)
        self.writer_HybridDDQN.add_scalar('Distance travelled/starting distance', travelled_distance_HybridDDQN/starting_distance_HybridDDQN, 0)

        self.writer_DQN.add_scalar('Direction changes/success', direction_changes_DQN/arrived_DQN, 0)
        self.writer_DDQN.add_scalar('Direction changes/success', direction_changes_DDQN/arrived_DDQN, 0)
        self.writer_HybridDQN.add_scalar('Direction changes/success', direction_changes_HybridDQN/arrived_HybridDQN, 0)
        self.writer_HybridDDQN.add_scalar('Direction changes/success', direction_changes_HybridDDQN/arrived_HybridDDQN, 0)

        self.writer_DQN.add_scalar('Time to arrive/successful arrival', total_arrive_time_DQN/arrived_DQN, 0)
        self.writer_DDQN.add_scalar('Time to arrive/successful arrival', total_arrive_time_DDQN/arrived_DDQN, 0)
        self.writer_HybridDQN.add_scalar('Time to arrive/successful arrival', total_arrive_time_HybridDQN/arrived_HybridDQN, 0)
        self.writer_HybridDDQN.add_scalar('Time to arrive/successful arrival', total_arrive_time_HybridDDQN/arrived_HybridDDQN, 0)    

        result = [[]]

        result[0].append(total_reward_DQN/(args.num_scenarios * args.num_agents))
        result[0].append(arrived_DQN/(args.num_scenarios * args.num_agents))
        result[0].append(travelled_distance_DQN/starting_distance_DQN)
        result[0].append(direction_changes_DQN/arrived_DQN)
        result[0].append(total_arrive_time_DQN/arrived_DQN)

        result.append([])
        result[1].append(total_reward_DDQN/(args.num_scenarios * args.num_agents))
        result[1].append(arrived_DDQN/(args.num_scenarios * args.num_agents))
        result[1].append(travelled_distance_DDQN/starting_distance_DDQN)
        result[1].append(direction_changes_DDQN/arrived_DDQN)
        result[1].append(total_arrive_time_DDQN/arrived_DDQN)

        result.append([])
        result[2].append(total_reward_HybridDQN/(args.num_scenarios * args.num_agents))
        result[2].append(arrived_HybridDQN/(args.num_scenarios * args.num_agents))
        result[2].append(travelled_distance_HybridDQN/starting_distance_HybridDQN)
        result[2].append(direction_changes_HybridDQN/arrived_HybridDQN)
        result[2].append(total_arrive_time_HybridDQN/arrived_HybridDQN)

        result.append([])
        result[3].append(total_reward_HybridDDQN/(args.num_scenarios * args.num_agents))
        result[3].append(arrived_HybridDDQN/(args.num_scenarios * args.num_agents))
        result[3].append(travelled_distance_HybridDDQN/starting_distance_HybridDDQN)
        result[3].append(direction_changes_HybridDDQN/arrived_HybridDDQN)
        result[3].append(total_arrive_time_HybridDDQN/arrived_HybridDDQN)

        df = pd.DataFrame(result, columns=['Reward per agent', 'Arrival rate', 'Real Path/Ideal Path', '# direction changes', 'Time to arrive'], index=['DQN', 'DDQN', 'Hybrid DQN', 'Hybrid DDQN'])
        df.to_csv(self.csv_path)

        self.writer_DQN.close()
        self.writer_DDQN.close()
        self.writer_HybridDQN.close()
        self.writer_HybridDDQN.close()

def main():
    trainer = Evaluator()
    trainer.evaluate()

if __name__ == "__main__":
    main()