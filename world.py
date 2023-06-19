# import numpy as np

# from agent import Agent

# class World:
#     def __init__(self, num_columns, num_rows, num_bots) -> None:
#         self.num_columns = num_columns
#         self.num_rows = num_rows
#         self.num_bots = num_bots

#         self.start_postitions = []
#         self.goal_positions = []
#         self.agent_lists = []
#         self._init_positions()

#     def _init_positions(self):
#         self.start_postitions.append(self._random_position())
#         num_initialized_bots = 1
#         while num_initialized_bots < self.num_bots:
#             new_position = self._random_position()
#             if new_position not in self.start_postitions:
#                 self.start_postitions.append(new_position)
#                 num_initialized_bots += 1

#         num_initialized_bots = 0
#         while num_initialized_bots < self.num_bots:
#             new_position = self._random_position()
#             if new_position not in self.start_postitions and new_position not in self.goal_positions:
#                 self.goal_positions.append(new_position)
#                 num_initialized_bots += 1

#         for i in range(self.num_bots):
#             self.agent_lists.append(Agent(self.start_postitions[i], self.goal_positions[i]))

#     def _random_position(self):
#         column = np.random.randint(0, self.num_columns)
#         row = np.random.randint(0, self.num_rows)

#         return (column, row)
    
# if __name__ == '__main__':
#     world = World(10, 12, 5)
#     print(world.start_postitions)
#     print(world.goal_positions)
