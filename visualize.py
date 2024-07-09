import pygame
import math
import random
import time
import numpy as np

from models.QNet import QNetwork
import torch
import pytorch_lightning as pl

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

        self.display = pygame.display.set_mode(
            (self.screen_width, self.screen_height))
        self._init_display()

        self.colors_list = [()] * num_boxes
        self.colors_list = list(
            map(lambda _: tuple(np.random.randint(0, 256, size=3)), self.colors_list))

    def _random_color(self,):
        return tuple(np.random.randint(0, 256, size=3))

    def _init_display(self):
        pygame.display.get_surface().fill(Visualize.Color.WHITE)  # background
        self._draw_grid()

    def _draw_grid(self):
        for x in range(0, self.screen_width, self.box_width):
            for y in range(0, self.screen_height, self.box_height):
                pygame.draw.rect(self.display, Visualize.Color.BLACK, [
                                 x, y, self.box_width, self.box_height], 1)

    def _draw_box(self, x, y, color, isSmall=False):  # x, y are indices in column and row
        if not isSmall:
            pygame.draw.rect(self.display, color, [
                             x * self.box_width, y * self.box_height, self.box_width, self.box_height])
        else:
            pygame.draw.rect(self.display, color, [x * self.box_width + self.box_width / 4, y *
                             self.box_height + self.box_height / 4, self.box_width / 2, self.box_height / 2])

    def _draw_goal(self, x, y, color):  # x, y are indices in column and row
        pygame.draw.ellipse(self.display, color, [
                            x * self.box_width, y * self.box_height, self.box_width, self.box_height])

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
            # column, row = agent_position
            column, row = (agent_position[0] - old_position[agent_idx][0]) * 2 + \
                old_position[agent_idx][0], (agent_position[1] -
                                             old_position[agent_idx][1]) * 2 + old_position[agent_idx][1]

            self._draw_box(column, row, color, True)

        for agent_idx, agent_goal in enumerate(world.goal_positions):
            color = self.colors_list[agent_idx]
            column, row = agent_goal
            self._draw_goal(column, row, color)

        for wall in world.wall_positions:
            column, row = wall
            self._draw_box(column, row, Visualize.Color.BLACK)

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
    visualizer = Visualize(800, 800, args.num_rows,
                           args.num_columns, args.num_agents)
    # model = DQNLightning.load_from_checkpoint(args.visualize_ckpt)

    model = QNetwork().to(torch.device("cpu"))
    # model.load_model('runs/DDQN/run_1/model.pt')
    model.load_state_dict(torch.load(
        'runs/DDQN/run_1/model.pt', map_location=torch.device('cpu')))
    # torch.load()
    model.eval()

    while True:
        world = World(args.num_rows, args.num_columns, args.num_agents)
        done = False
        agents_done = [False] * args.num_agents
        frequent = world.frequent
        counter = 0
        old_actions = [0] * args.num_agents
        while not done:
            old_position = world.current_positions.copy()
            for agent_idx in range(world.num_bots):
                if agents_done[agent_idx]:
                    continue
                agent = world.agent_lists[agent_idx]
                state = agent.get_state()

                if counter % 4 != 0:
                    action = old_actions[agent_idx]
                    reward, agent_done, arrived = world.perform_action(
                        action, agent_idx, 0.25)
                    if agent_done:
                        agents_done[agent_idx] = True
                    continue

                # state = torch.tensor(state)
                predictions = model(state)

                sorted_values, sorted_actions = torch.sort(
                    predictions, descending=True)

                value_actions = list(zip(sorted_values, sorted_actions))
                i, j = world.current_positions[agent_idx]
                i = round(i)
                j = round(j)

                tmp = 10
                # print(frequent[i][j])
                frequent[i][j] += 1

                for tup_idx in range(len(value_actions)):
                    val, action = value_actions[tup_idx]
                    if not world.is_valid_action(action, agent_idx):
                        continue
                    if action == Action.STATIC:
                        val = val * (1 - tmp * frequent[i][j] / 100.0)
                    elif action == Action.LEFT:
                        val = val * (1 - tmp * frequent[i - 1][j] / 100.0)
                    elif action == Action.DOWN:
                        val = val * (1 - tmp * frequent[i][j + 1] / 100.0)
                    elif action == Action.RIGHT:
                        val = val * (1 - tmp * frequent[i + 1][j] / 100.0)
                    elif action == Action.UP:
                        val = val * (1 - tmp * frequent[i][j - 1] / 100.0)

                    value_actions[tup_idx] = val, action

                value_actions.sort(reverse=True, key=lambda x: x[0])

                # print(value_actions[0][0], value_actions[1][0], value_actions[2][0], value_actions[3][0], value_actions[4][0])
                # print(value_actions[0][1], value_actions[1][1], value_actions[2][1], value_actions[3][1], value_actions[4][1])

                sorted_actions = list(map(lambda x: x[1], value_actions))

                performed_the_move = False
                action = sorted_actions[0]
                for idx in range(len(sorted_actions)):
                    action = sorted_actions[idx]

                    if world.is_valid_action(action, agent_idx):
                        reward, agent_done, arrived = world.perform_action(
                            action, agent_idx, 0.25)
                        if agent_done:
                            agents_done[agent_idx] = True
                        performed_the_move = True
                        break

                old_actions[agent_idx] = action

                if not performed_the_move:
                    raise RuntimeError("Cannot perform any move!")
            visualizer.draw_state(world, old_position)

            if all(agents_done):
                pygame.time.wait(500)
                old_position = world.current_positions
                visualizer.draw_state(world, old_position)
                pygame.time.wait(200)
                done = True
                break
            else:
                pygame.time.wait(200)

            counter += 1


if __name__ == '__main__':
    main()
