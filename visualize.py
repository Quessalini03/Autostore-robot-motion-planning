import pygame
import math, random
import time
import numpy as np

import torch
import pytorch_lightning as pl

import matplotlib.pyplot as plt

from lightning_model import DQNLightning
from config import Action, args
from agent import World

class Visualize:
    class Color():
        BLACK = (0, 0, 0)
        WHITE = (200, 200, 200)

    def __init__(self, screen_width, screen_height, num_columns, num_rows, num_boxes):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.num_boxes = num_boxes

        self.box_width = math.floor(screen_width / num_columns)
        self.box_height = math.floor(screen_height / num_rows)

        self.display = pygame.display.set_mode((self.screen_width, self.screen_height))
        self._init_display()
        
        self.colors_list = [()] * num_boxes
        self.colors_list = list(map(lambda _: tuple(np.random.randint(0, 256, size=3)), self.colors_list))
    
    def _random_color(self,):
        return tuple(np.random.randint(0, 256, size=3))
    
    def _init_display(self):
        pygame.display.get_surface().fill(Visualize.Color.WHITE)  # background
        self._draw_grid()

    def _draw_grid(self):
        for x in range(0, self.screen_width, self.box_width):
            for y in range(0, self.screen_height, self.box_height):
                pygame.draw.rect(self.display, Visualize.Color.BLACK, [x, y, self.box_width, self.box_height], 1)
    
    def _draw_box(self, x, y, color, isSmall = False): # x, y are indices in column and row
        if not isSmall: pygame.draw.rect(self.display, color, [x * self.box_width, y * self.box_height, self.box_width, self.box_height])
        else: pygame.draw.rect(self.display, color, [x * self.box_width + self.box_width / 4, y * self.box_height + self.box_height / 4, self.box_width / 2, self.box_height / 2])

    def _draw_goal(self, x, y, color): # x, y are indices in column and row
        pygame.draw.ellipse(self.display, color, [x * self.box_width, y * self.box_height, self.box_width, self.box_height])

    def _update_display(self):
        pygame.display.update()

    def draw_state(self, world: World, old_position):
        for column in range(self.num_columns):
            for row in range(self.num_rows):
                    self._draw_box(column, row, Visualize.Color.WHITE)

        for agent_idx, agent_position in enumerate(old_position):
            color = self.colors_list[agent_idx]
            column, row = agent_position
            self._draw_box(column, row, color)
        
        for agent_idx, agent_position in enumerate(world.current_positions):
            color = self.colors_list[agent_idx]
            column, row = agent_position
            self._draw_box(column, row, color, True)
        
        for agent_idx, agent_goal in enumerate(world.goal_positions):
            color = self.colors_list[agent_idx]
            column, row = agent_goal
            self._draw_goal(column, row, color)

        # y = 0
        # for row in state_matrix:
        #     x = 0
        #     for item in row:
        #         if item == 0:
        #             self._draw_box(x, y, Visualize.Color.WHITE)
        #         else:
        #             color = self.colors_list[abs(item) - 1]
        #             if item > 0: # is box
        #                 self._draw_box(x, y, color)
        #             else: # is goal
        #                 self._draw_goal(x, y, color)
        #         x += self.box_width
        #     y += self.box_height

        self._draw_grid()
        self._update_display()

    def test(self):
        self._draw_goal(0, 0, (156, 100, 78))
        self._update_display()

class DataPlot:
    ideal_lengths = []
    real_lengths = []
    time = []

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def main():
    visualizer = Visualize(800, 800, args.num_rows, args.num_columns, args.num_agents)
    model = DQNLightning.load_from_checkpoint(args.visualize_ckpt)
    model.eval()
    
    for i in range(args.num_agents):
        DataPlot.ideal_lengths.append([])
        DataPlot.real_lengths.append([])
        DataPlot.time.append(0)

    isRunnning = True
    while isRunnning:
        
        world = World(args.num_rows, args.num_columns, args.num_agents)
        done = False
        agents_done = [False] * args.num_agents
        frequent = world.frequent
        
        for agent_idx in range(args.num_agents):
            x1, y1 = world.current_positions[agent_idx]
            x2, y2 = world.goal_positions[agent_idx]
            d = abs(x2 - x1) + abs(y2 - y1)
            DataPlot.ideal_lengths[agent_idx].append(d)
        
        Time = 0    
        
        while not done and isRunnning:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    isRunnning = False
            Time += 1
            old_position = world.current_positions.copy()
            for agent_idx in range(world.num_bots):
                if agents_done[agent_idx]:
                    if len(DataPlot.ideal_lengths[agent_idx]) > len(DataPlot.real_lengths[agent_idx]):
                        DataPlot.real_lengths[agent_idx].append(Time)
                    continue
                agent = world.agent_lists[agent_idx]
                state = agent.get_state()
                state = torch.tensor(state)
                predictions = model(state)

                sorted_values, sorted_actions = torch.sort(predictions, descending=True)
                
                value_actions = list(zip(sorted_values, sorted_actions))
                i, j = world.current_positions[agent_idx]
                
                percent_degree = 10
                #print(frequent[agent_idx][i][j])
                
                for tup_idx in range(len(value_actions)):
                    val, action = value_actions[tup_idx]
                    if not world.is_valid_action(action, agent_idx):
                        continue
                    if action == Action.STATIC:
                        val = val * (1 - percent_degree * frequent[agent_idx][i][j] / 100.0)
                    elif action == Action.LEFT:
                        val = val * (1 - percent_degree * frequent[agent_idx][i - 1][j] / 100.0)
                    elif action == Action.DOWN:
                        val = val * (1 - percent_degree * frequent[agent_idx][i][j + 1] / 100.0)
                    elif action == Action.RIGHT:
                        val = val * (1 - percent_degree * frequent[agent_idx][i + 1][j] / 100.0)
                    elif action == Action.UP:
                        val = val * (1 - percent_degree * frequent[agent_idx][i][j - 1] / 100.0)
                        
                    value_actions[tup_idx] = val, action
                    
                value_actions.sort(reverse=True, key = lambda x: x[0])
                    
                sorted_actions = list(map(lambda x: x[1], value_actions))
                
                performed_the_move = False
                for idx in range(len(sorted_actions)):
                    action = sorted_actions[idx]
                                        
                    if world.is_valid_action(action, agent_idx):
                        _, agent_done = world.perform_action(action, agent_idx)
                        if agent_done:
                            agents_done[agent_idx] = True
                        performed_the_move = True
                        break

                if not performed_the_move:
                    raise RuntimeError("Cannot perform any move!")
            visualizer.draw_state(world, old_position)
            
            if all(agents_done):
                pygame.time.wait(100)
                old_position = world.current_positions
                visualizer.draw_state(world, old_position)
                pygame.time.wait(100)
                done = True
                break
            else:
                pygame.time.wait(100)
            
    print(DataPlot.ideal_lengths)    
    print(DataPlot.real_lengths)    
    
    for i in range(args.num_agents):
        if len(DataPlot.ideal_lengths[agent_idx]) > len(DataPlot.real_lengths[agent_idx]):
            DataPlot.ideal_lengths[i].pop()

    DataPlot.ideal_lengths = flatten_list(DataPlot.ideal_lengths)
    DataPlot.real_lengths = flatten_list(DataPlot.real_lengths)

    DataPlot.real_lengths = list(map(lambda x: round(x), DataPlot.real_lengths))
    print(DataPlot.ideal_lengths)
    print(DataPlot.real_lengths)

    val = []
    temp = []

    for i in range(len(DataPlot.ideal_lengths)):
        if DataPlot.ideal_lengths[i] == 0:
            DataPlot.ideal_lengths[i] = 1
        val.append(DataPlot.real_lengths[i] / DataPlot.ideal_lengths[i])

    sum_val = 0
    y3 = []
    for i in range(len(val)):
        sum_val += val[i]
        y3.append(sum_val / (i + 1))

    plt.plot(range(len(val)), y3, marker='o', linestyle='-', color='red', label='Rate')

    plt.xlabel('ith completions')
    plt.ylabel('Average ratio between real path and ideal path')

    plt.title('Ratio graph between actual and ideal path')

    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()