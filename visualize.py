import pygame
import math
import numpy as np


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
    
    def _draw_box(self, x, y, color): # x, y are the center
        pygame.draw.rect(self.display, color, [x - self.box_width/2, y - self.box_height/2, self.box_width, self.box_height])

    def _draw_goal(self, x, y, color): # x, y are the center
        pygame.draw.ellipse(self.display, color, [x - self.box_width/2, y - self.box_height/2, self.box_width, self.box_height])

    def _update_display(self):
        pygame.display.update()

    def draw_state(self, state_matrix):
        y = self.box_height / 2
        for row in state_matrix:
            x = self.box_width / 2
            for item in row:
                if item == 0:
                    self._draw_box(x, y, Visualize.Color.WHITE)
                else:
                    color = self.colors_list[abs(item) - 1]
                    if item > 0: # is box
                        self._draw_box(x, y, color)
                    else: # is goal
                        self._draw_goal(x, y, color)
                x += self.box_width
            y += self.box_height

        self._draw_grid()
        self._update_display()

    def test(self):
        self._draw_goal(self.box_width/2, self.box_height/2, (156, 100, 78))
        self._update_display()


if __name__ == '__main__':
    visualize = Visualize(800, 800, 4, 4, 4)
    state = [[1 ,0 , 0, -1],
            [2 ,0 , 0, -4],
            [3 ,0 , 0, -3],
            [4 ,0 , 0, -2]]
    visualize.draw_state(state)
    while True:
        pass