import torch
from typing import Tuple
import numpy as np
from copy import deepcopy

from config import *
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

    def find_current_position(self, position: Tuple[int, int]):
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


class Agent:
    net = DQN(65, 5, 256)
    target_net = DQN(61, 5, 256).load_state_dict(net.state_dict())

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

                other_agent_index = self.world.find_current_position((current_col + col_offset, current_row + row_offset))
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
        

    def get_action(self, state):
        move = [0, 0, 0, 0, 0]
        if np.random.random() < self.epsilon:
            move_index = np.random.randint(0, self.num_actions)
            move[move_index] = 1
        else:
            current_state = torch.tensor(state, dtype=torch.float32)
            prediction = Agent.net(current_state)
            move_index = torch.argmax(prediction).item()
            move[move_index] = 1

        return move
        

if __name__ == '__main__':
    world = World(3, 3, 3)
    print(world.start_postitions)
    print(world.goal_positions)
    print(world.agent_lists[0].get_state())
    print(world.agent_lists[1].get_state())
    print(world.agent_lists[2].get_state())
    print(len(world.agent_lists[2].get_state()))
