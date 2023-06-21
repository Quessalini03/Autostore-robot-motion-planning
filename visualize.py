import pygame
import math
import time
import numpy as np

import torch
import pytorch_lightning as pl

from lightning_model import DQNLightning
from config import args
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
    
    def _draw_box(self, x, y, color): # x, y are indices in column and row
        pygame.draw.rect(self.display, color, [x * self.box_width, y * self.box_height, self.box_width, self.box_height])

    def _draw_goal(self, x, y, color): # x, y are indices in column and row
        pygame.draw.ellipse(self.display, color, [x * self.box_width, y * self.box_height, self.box_width, self.box_height])

    def _update_display(self):
        pygame.display.update()

    def draw_state(self, world: World):
        for column in range(self.num_columns):
            for row in range(self.num_rows):
                    self._draw_box(column, row, Visualize.Color.WHITE)

        for agent_idx, agent_position in enumerate(world.current_positions):
            color = self.colors_list[agent_idx]
            column, row = agent_position
            self._draw_box(column, row, color)

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


def main():
    visualizer = Visualize(800, 800, 7, 7, args.num_agents)
    model = DQNLightning.load_from_checkpoint(args.visualize_ckpt)
    model.eval()

    while True:
        world = World(7, 7, args.num_agents)
        done = False
        agents_done = [False] * args.num_agents
        while not done:
            visualizer.draw_state(world)
            for agent_idx in range(world.num_bots):
                if agents_done[agent_idx]:
                    continue
                agent = world.agent_lists[agent_idx]
                state = agent.get_state()
                state = torch.tensor(state)
                predictions = model(state)
                _, sorted_actions = torch.sort(predictions, descending=True)

                performed_the_move = False
                for action in sorted_actions:
                    if world.is_valid_action(action, agent_idx):
                        _, agent_done = world.perform_action(action, agent_idx)
                        if agent_done:
                            agents_done[agent_idx] = True
                        performed_the_move = True
                        break

                if not performed_the_move:
                    raise RuntimeError("Cannot perform any move!")
                
            if all(agents_done):
                pygame.time.wait(500)
                visualizer.draw_state(world)
                pygame.time.wait(5000)
                done = True
                break
            else:
                pygame.time.wait(500)


if __name__ == '__main__':
    main()