import torch
from tqdm import tqdm
import time
import pandas as pd

from models.QNet import QNetwork
from agent import World, Agent
from evaluator.BaseEvaluator import BaseEvaluator

from torch.utils.tensorboard import SummaryWriter


class args:
    num_columns = 22
    num_rows = 22
    num_agents = 70

    num_scenarios = 200
    obstacle_matrix_1 = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,],
        [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,],
        [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,],
        [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
    ]

    obstacle_matrix_2 = None
    obstacle_matrix = obstacle_matrix_2

class Evaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__('ML_assignment')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net_DDQN = QNetwork().to(self.device)
        self.policy_net_DDQN.load_model('runs/DDQN/run_2/model.pt')
        self.policy_net_DDQN.eval()

        self.writer_DDQN = SummaryWriter(self.get_eval_directory() + '/DDQN')

        self.csv_path = self.get_eval_directory() + '/result.csv'

        self.save_evaluate_params(args)

    def evaluate(self):
        world = World(args.num_columns, args.num_rows, args.num_agents, args.obstacle_matrix)
        world.eval()
        progess_bar = tqdm(total=args.num_scenarios, desc="Scenarios completed", leave=True, unit_scale=True, unit='Scenario', colour='green')


        total_reward_DDQN = 0

        arrived_DDQN = 0

        starting_distance_DDQN = 0
        travelled_distance_DDQN = 0
        direction_changes_DDQN = 0
        total_arrive_time_DDQN = 0
        simulation_time_DDQN = 0

        for scenario in range(args.num_scenarios):
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

            end_time = time.time()
            simulation_time_DDQN += end_time - start_time

            self.writer_DDQN.add_scalar('Arrival rate/scenario', arrived_at_goal/args.num_agents, scenario)
            self.writer_DDQN.add_scalar('Reward/scenario', episode_reward, scenario)
            self.writer_DDQN.add_scalar('Total reward', total_reward_DDQN, scenario)


            # hard reset world
            world = World(args.num_columns, args.num_rows, args.num_agents, args.obstacle_matrix)
            world.eval()

            progess_bar.update(1)


        self.writer_DDQN.add_scalar('Total reward/scenario', total_reward_DDQN/args.num_scenarios, 0)
        self.writer_DDQN.add_scalar('Arrival rate/total', arrived_DDQN/(args.num_scenarios * args.num_agents), 0)
        self.writer_DDQN.add_scalar('Distance travelled/starting distance', travelled_distance_DDQN/starting_distance_DDQN, 0)
        self.writer_DDQN.add_scalar('Direction changes/success', direction_changes_DDQN/arrived_DDQN, 0)
        self.writer_DDQN.add_scalar('Time to arrive/successful arrival', total_arrive_time_DDQN/arrived_DDQN, 0)
        self.writer_DDQN.add_scalar('Simulation time/scenario', simulation_time_DDQN/args.num_scenarios, 0)


        result = [[]]

        result[0].append(total_reward_DDQN/(args.num_scenarios * args.num_agents))
        result[0].append(arrived_DDQN/(args.num_scenarios * args.num_agents))
        result[0].append(travelled_distance_DDQN/starting_distance_DDQN)
        result[0].append(direction_changes_DDQN/arrived_DDQN)
        result[0].append(total_arrive_time_DDQN/arrived_DDQN)
        result[0].append(simulation_time_DDQN/args.num_scenarios)


        df = pd.DataFrame(result, columns=['Reward per agent', 'Arrival rate', 'Real Path/Ideal Path', '# direction changes/success', 'Time to arrive', 'Simulation time/scenario'], index=['DDQN'])
        df.to_csv(self.csv_path)

        self.writer_DDQN.close()

def main():
    trainer = Evaluator()
    trainer.evaluate()

if __name__ == "__main__":
    main()