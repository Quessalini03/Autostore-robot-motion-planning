import torch
from typing import Tuple
import numpy as np
from copy import deepcopy
from collections import deque

import torch.nn as nn

from config import args, Action
from models.QNet import QNetwork, QLoss, ReplayMemory, Experience

class World:
    def __init__(self, num_columns, num_rows, num_bots) -> None:
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.num_bots = num_bots

        self.wall_positions = []
        self.start_postitions = []
        self.current_positions = []
        self.goal_positions = []
        self.agent_lists = []
        self.num_actions_performed = [0 for _ in range(self.num_bots)]
        self._init_positions()
        self.frequent = []
        for i in range(num_rows):
            self.frequent.append([])
            for j in range(num_columns):
                self.frequent[i].append(0)

    def _init_positions(self):
        # TODO: add walls
        for i in range(1, self.num_rows - 1):
            self.wall_positions.append((0, i))
            self.wall_positions.append((self.num_columns - 1, i))
        
        for i in range(self.num_columns):
            self.wall_positions.append((i, 0))
            self.wall_positions.append((i, self.num_rows - 1))

        num_initialized_bots = 0
        while num_initialized_bots < self.num_bots:
            new_position = self._random_position()
            if new_position not in self.start_postitions and new_position not in self.wall_positions:
                self.start_postitions.append(new_position)
                num_initialized_bots += 1

        self.current_positions = deepcopy(self.start_postitions)

        num_initialized_bots = 0
        while num_initialized_bots < self.num_bots:
            new_position = self._random_position()
            if new_position not in self.start_postitions and new_position not in self.goal_positions and new_position not in self.wall_positions:
                self.goal_positions.append(new_position)
                num_initialized_bots += 1

        for i in range(self.num_bots):
            self.agent_lists.append(Agent(self, i))
        self.frequent = []
        for i in range(self.num_rows):
            self.frequent.append([])
            for j in range(self.num_columns):
                self.frequent[i].append(0)    

    def _random_position(self):
        column = np.random.randint(0, self.num_columns)
        row = np.random.randint(0, self.num_rows)

        return (column, row)
    
    def is_valid_position(self, position, column_offset, row_offset):
        current_col, current_row = position
        position = (current_col + column_offset, current_row + row_offset)

        if position in self.wall_positions:
            return False
        
        if position[0] >= self.num_columns or \
            position[1] >= self.num_rows or \
            position[0] < 0 or \
            position[1] < 0:
                return False
        
        return True

    def current_position_is_of(self, position: Tuple[int, int]):
        for i in range(self.num_bots):
            if self.agent_lists[i].is_alive == False:
                continue
            if self.current_positions[i] == position:
                return i
        
        return -1
    
    def goal_position_of(self, position: Tuple[int, int]):
        for i in range(self.num_bots):
            if self.agent_lists[i].is_alive == False:
                continue
            if self.goal_positions[i] == position:
                return i
        
        return -1
    
    def _move(self, action: int, agent_index: int):
        cur_col, cur_row = self.current_positions[agent_index]
        if action == Action.UP:
            cur_row -= 1
        elif action == Action.DOWN:
            cur_row += 1
        elif action == Action.LEFT:
            cur_col -= 1
        elif action == Action.RIGHT:
            cur_col += 1
        
        return (cur_col, cur_row)

    def _move_inplace(self, action: int, agent_index: int):
        cur_col, cur_row = self._move(action, agent_index)

        self.current_positions[agent_index] = (cur_col, cur_row)

    def _is_collided(self, agent_index: int):
        # collide with walls
        current_position = self.current_positions[agent_index]
        if current_position in self.wall_positions:
            return True
        
        # collide with other living agents
        for i in range(self.num_bots):
            if i == agent_index:
                continue
            if self.agent_lists[i].is_alive == False:
                continue
            if self.current_positions[i] == current_position:
                return True
            
        return False


    
    def perform_action(self, action: int, agent_index: int):
        # self.frequent[self.current_positions[agent_index][0]][self.current_positions[agent_index][1]] += 1.0
        self._move_inplace(action, agent_index)

        done = 0
        reward = 0.0
        if self._is_collided(agent_index):
            done = 1
            reward = -2
            self.agent_lists[agent_index].is_alive = False
            return reward, done
        
        agent = self.agent_lists[agent_index]
        agent.time_to_live -= 1
        if agent.time_to_live <= 0:
            done = 1
            reward = -0.01
            self.agent_lists[agent_index].is_alive = False
            return reward, done
        
        if self.current_positions[agent_index] == self.goal_positions[agent_index]:
            done = 1
            reward = 2
            # self.agent_lists[agent_index].is_alive = False
            return reward, done
        
        reward = -0.01
        done = 0
        return reward, done

    def is_valid_action(self, action: int, agent_index: int,):
        current_position = self._move(action, agent_index)
        curr_column, curr_row = current_position

        if curr_column >= self.num_columns or \
           curr_row >= self.num_rows or \
           curr_column < 0 or \
           curr_row < 0:
            return False
        
        for i in range(self.num_bots):
            if i == agent_index:
                continue
            if self.agent_lists[i].is_alive == False:
                continue
            if self.current_positions[i] == current_position:
                return False
            
        return True

    



class Agent:
    epsilon = args.epsilon
    epsilon_decrement = args.epsilon_decrement
    gamma = args.gamma

    def __init__(
        self,
        world: World,
        index_in_world: int,
        num_actions = args.num_actions,
        time_to_live = args.patient_factor
    ):
        self.world = world
        self.index_in_world = index_in_world
        self.num_actions = num_actions

        self.is_alive = True
        self.time_to_live = time_to_live
        # self.previous_move = 4 # static

        self.default_state = [[0, 0, 0, 0, 0] for _ in range(5)]
        self.observation_history = deque([self.default_state for _ in range(3)], maxlen=3)
        self.is_first_observation = True

    def reset(self):
        self.is_alive = True
        self.observation_history = deque([self.default_state for _ in range(3)], maxlen=3)
        self.is_first_observation = True

    def get_state(self):
        current_position = self.world.current_positions[self.index_in_world]
        goal_position = self.world.goal_positions[self.index_in_world]
        
        current_col, current_row = current_position
        # check surrounding regions: [valid_position, other_agent_present, prev_move_up, prev_move_down, prev_move_left, prev_move_right, prev_move_static] * 8
        # 5*5 center
        center = (2, 2)
        observation = [[0, 0, 0, 0, 0] for _ in range(5)]
        for row_offset in range(-2, 3):
            for col_offset in range(-2, 3):
                if row_offset == 0 and col_offset == 0:
                    continue
                if not self.world.is_valid_position(current_position, col_offset, row_offset):
                    observation[center[0] + col_offset][center[1] + row_offset] = 1
                    # print("Position: ", (current_position[0] + col_offset, current_position[1] + row_offset), " is not valid")
                    continue

                other_agent_index = self.world.current_position_is_of((current_col + col_offset, current_row + row_offset))
                if other_agent_index != -1: # another agent is at this position
                    observation[center[0] + col_offset][center[1] + row_offset] = 1
                    # print("Position: ", (current_position[0] + col_offset, current_position[1] + row_offset), " is occupied by another agent")
                    continue

                # print("Position: ", (current_position[0] + col_offset, current_position[1] + row_offset), " is valid")
        
        if self.is_first_observation:
            self.is_first_observation = False
            self.observation_history = deque([observation for _ in range(3)], maxlen=3)
        else:
            self.observation_history.append(observation)

        goal_col, goal_row = goal_position
        goal_distance = goal_col - current_col, goal_row - current_row
            
        state = {
            "observation": torch.tensor(self.observation_history, dtype=torch.float32),
            "goal": torch.tensor(goal_distance, dtype=torch.float32),
        }

        return state
        
    def get_epsilon(self):
        return_val = Agent.epsilon
        if Agent.epsilon > 0.1:
            Agent.epsilon = Agent.epsilon - Agent.epsilon_decrement
        return return_val

    def get_action(self, policy_net: QNetwork, state):
        if np.random.random() < self.get_epsilon():
            move_index = np.random.randint(0, self.num_actions)
        else:
            prediction = policy_net.forward(state)
            move_index = torch.argmax(prediction).item()

        return move_index
    
    def perform_action(self, action: int):
        self.previous_move = action
        return self.world.perform_action(action, self.index_in_world)
    
    @torch.no_grad()
    def play_step(self, net: nn.Module, buffer: ReplayMemory):
        current_state = self.get_state()
        action = self.get_action(net, current_state)

        reward, done = self.perform_action(action)
        new_state = self.get_state()
        exp = Experience(current_state, action, reward, new_state, done)

        buffer.append(exp)

        return reward, done
        

if __name__ == '__main__':
    world = World(5, 5, 3)
    print(world.start_postitions)
    print(world.goal_positions)
    print(world.wall_positions)
    print("Agent 1:")
    print(world.agent_lists[0].get_state())
    print("Agent 2:")
    print(world.agent_lists[1].get_state())
    print("Agent 3:")
    print(world.agent_lists[2].get_state())
    print(len(world.agent_lists[2].get_state()))
