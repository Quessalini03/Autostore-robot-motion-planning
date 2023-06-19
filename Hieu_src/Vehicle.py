import random
import numpy as np

Width = 40
#walk_r = -0.04
actions = ["up", "down", "left", "right", "stay"]
[x, y] = [3, 3]
score = 0
reward = 0
action = 0
s_ = [0, 0]

class Vehicle:
    def __init__(self,
        id,
    	coord, #start position
    	score,
        board_id,
        t_id,
        dest, #goal
        board_id_d,
        board_id_d_t,
        reward):
        self.id = id
        self.coord = coord
        self.score = score
        self.board_id = board_id
        self.t_id = t_id
        self.dest = dest
        self.board_id_d = board_id_d
        self.board_id_d_t = board_id_d_t 
        self.steps_od = abs(dest[0] - coord[0]) + abs(dest[1] - coord[1])
        self.reward = reward

    
    def set_reward(self, rewards):
        self.rewards = rewards
        self.score += rewards

    def get_reward(self):
        return self.reward

    def get_action(self):
        return self.action

    def get_dest(self):
        return self.dest

    def set_s_(self, s_):
        self.s_ = s_

    def get_id(self):
        return self.id

    def get_coord(self):
        return self.coord

    def get_board_id(self):
        return self.get_board_id

    def set_board_id(self, board_id):
        self.board_id = board_id

    def set_t_id(self, t_id):
        self.t_id = t_id
