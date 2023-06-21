import torch
from typing import Tuple
import numpy as np
from copy import deepcopy

import torch.nn as nn

from config import args, Action
from buffer import ReplayBuffer, Experience
from net import DQN

class World:
    def __init__(self, num_columns, num_rows, num_bots) -> None:
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.num_bots = num_bots

        self.start_postitions = []
        self.current_positions = []
        self.goal_positions = []
        self.agent_lists = []
        self.num_actions_performed = [0] * self.num_bots
        self._init_positions()

    def _init_positions(self):
        self.start_postitions.append(self._random_position())
        num_initialized_bots = 1
        while num_initialized_bots < self.num_bots:
            new_position = self._random_position()
            if new_position not in self.start_postitions:
                self.start_postitions.append(new_position)
                num_initialized_bots += 1

        self.current_positions = deepcopy(self.start_postitions)

        num_initialized_bots = 0
        while num_initialized_bots < self.num_bots:
            new_position = self._random_position()
            if new_position not in self.start_postitions and new_position not in self.goal_positions:
                self.goal_positions.append(new_position)
                num_initialized_bots += 1

        for i in range(self.num_bots):
            self.agent_lists.append(Agent(self, i))

    def _random_position(self):
        column = np.random.randint(0, self.num_columns)
        row = np.random.randint(0, self.num_rows)

        return (column, row)
    
    def is_valid_position(self, position, column_offset, row_offset):
        current_col, current_row = position

        if current_col + column_offset < 0 or current_col + column_offset >= self.num_columns:
            return False
        
        if current_row + row_offset < 0 or current_row + row_offset >= self.num_rows:
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

        self.current_positions[agent_index] = (cur_col, cur_row)

    def _is_collided(self, agent_index: int):
        # collide with walls
        current_position = self.current_positions[agent_index]
        cur_col, cur_row = current_position
        if  cur_col >= self.num_columns or \
            cur_row >= self.num_rows or \
            cur_col < 0 or \
            cur_row < 0:
                return True
        
        # collide with other lving agents
        for i in range(self.num_bots):
            if i == agent_index:
                continue
            if self.agent_lists[i].is_alive == False:
                continue
            if self.current_positions[i] == current_position:
                return True
            
        return False


    
    def perform_action(self, action: int, agent_index: int):
        self._move(action, agent_index)

        done = 0
        reward = 0.0
        if self._is_collided(agent_index) or \
           self.num_actions_performed[agent_index] >= (self.num_columns * self.num_rows + args.patient_factor * (self.num_bots - 1)):
            done = 1
            reward = -10.0
            self.agent_lists[agent_index].is_alive = False
            return reward, done
        
        if self.current_positions[agent_index] == self.goal_positions[agent_index]:
            done = 1
            reward = 10.0
            self.agent_lists[agent_index].is_alive = False
            return reward, done
        
        reward = -0.05
        done = 0
        return reward, done

        

    



class Agent:
    def __init__(
        self,
        world: World,
        index_in_world: int,
        epsilon = args.epsilon,
        gamma = args.gamma,
        epsilon_decrement = args.epsilon_decrement,
        num_actions = args.num_actions
    ):
        self.world = world
        self.index_in_world = index_in_world
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decrement = epsilon_decrement
        self.num_actions = num_actions

        self.is_alive = True
        self.previous_move = 4 # static

    def get_state(self):
        return_state = []
        current_position = self.world.current_positions[self.index_in_world]
        goal_position = self.world.goal_positions[self.index_in_world]
        
        current_col, current_row = current_position
        # check surrounding regions: [valid_position, other_agent_present, prev_move_up, prev_move_down, prev_move_left, prev_move_right, prev_move_static] * 8
        for row_offset in range(-1, 2):
            for col_offset in range(-1, 2):
                if row_offset == 0 and col_offset == 0:
                    continue
                if not self.world.is_valid_position(current_position, col_offset, row_offset):
                    return_state += [False] * 7
                    continue

                other_agent_index = self.world.current_position_is_of((current_col + col_offset, current_row + row_offset))
                if other_agent_index == -1: # no agent is at this position
                    return_state += [True] + [False] * 6
                else:
                    other_agent_state_previous_move = [False] * 5
                    other_agent_state_previous_move[self.world.agent_lists[other_agent_index].previous_move] = True
                    return_state += [True] * 2 + other_agent_state_previous_move
        
        # goal position: [up, down, left, right] relative to the agent
        goal_col, goal_row = goal_position
        goal_state = [
            goal_row < current_row,
            goal_row > current_row,
            goal_col < current_col,
            goal_col > current_col,
        ]
        return_state += goal_state

        # TODO: blocking others goal: [is_blocking, agent_is_above, agent_is_below, agent_is_left, agent_is_right]
        position_is_goal_of_index = self.world.goal_position_of(current_position)
        if position_is_goal_of_index == -1 or position_is_goal_of_index == self.index_in_world:
            return_state += [False] * 5
        else:
            blocking_agent_state = [False] * 4
            blocking_agent_position = self.world.current_positions[position_is_goal_of_index]
            diff_in_col, diff_in_row = current_position[0] - blocking_agent_position[0], current_position[1] - blocking_agent_position[1]
            
            if diff_in_col == -1 and diff_in_row == 0:
                blocking_agent_state[3] = True
            elif diff_in_col == 1 and diff_in_row == 0:
                blocking_agent_state[2] = True
            elif diff_in_col == 0 and diff_in_row == -1:
                blocking_agent_state[1] = True
            elif diff_in_col == 0 and diff_in_row == 1:
                blocking_agent_state[0] = True

            return_state += [True] + blocking_agent_state
            

        return np.array(return_state, dtype=int)
        
    def get_epsilon(self):
        return_val = self.epsilon
        self.epsilon -= self.epsilon_decrement
        return return_val

    def get_action(self, net, state):
        if np.random.random() < self.get_epsilon():
            move_index = np.random.randint(0, self.num_actions)
        else:
            current_state = torch.tensor(state, dtype=torch.float32)
            prediction = net(current_state)
            move_index = torch.argmax(prediction).item()

        return move_index
    
    def perform_action(self, action: int):
        self.previous_move = action
        return self.world.perform_action(action, self.index_in_world)
    
    @torch.no_grad()
    def play_step(self, net: nn.Module, buffer: ReplayBuffer):
        current_state = self.get_state()
        action = self.get_action(net, current_state)

        reward, done = self.perform_action(action)
        new_state = self.get_state()
        exp = Experience(current_state, action, reward, done, new_state)

        buffer.append(exp)

        return reward, done
        

if __name__ == '__main__':
    world = World(3, 3, 3)
    print(world.start_postitions)
    print(world.goal_positions)
    print(world.agent_lists[0].get_state())
    print(world.agent_lists[1].get_state())
    print(world.agent_lists[2].get_state())
    print(len(world.agent_lists[2].get_state()))
