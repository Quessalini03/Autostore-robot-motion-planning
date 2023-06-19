"""
DeepQNetwork for route traffic guidance
Using Tkinter
"""
from tkinter import *
master = Tk()
import threading
import time
from Vehicle import Vehicle
from DQN import DeepQNetwork
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

Width = 40 #px size of each block of the grid
(x, y) = (10, 10) #Grid size
starting_agents = 80
board = Canvas(master, width=x*Width, height=y*Width) 
agents = []
walk_reward = 1
n_features = 104
n_actions = 2
walls = []
scores = []
vehicle_count = 1
origins = [[3, 3], [1, 3], [2, 1], [2, 3], [6, 9],
            [3, 5], [6, 7], [5, 3], [8, 6], [3, 9], 
            [2, 0], [5, 3], [2, 0], [9, 5], [2, 9], 
            [2, 3], [5, 6], [5, 0], [9, 1], [2, 5]]
destinations = [[1, 1], [5, 7], [1, 7], [5, 2], [9, 1],
            [2, 3], [8, 8], [2, 3], [1, 5], [5, 0],
            [9, 1], [2, 5], [8, 6], [3, 9], [6, 9],
            [3, 4], [5, 9], [6, 6], [7, 5], [2, 7]]
list_origins = 0
list_dests = 0

def render_grid():
    global walls, Width, x, y
    for i in range(x):
        for j in range(y):
            board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="white", width=1)
    for (i, j) in walls:
        board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="black", width=1)
render_grid()

def random_coord():
    new_x = random.randint(0, x-1) 
    new_y = random.randint(0, y-1)
    while ((new_x, new_y) in walls):
        new_x = random.randint(0, x-1) 
        new_y = random.randint(0, y-1)
    return [new_x, new_y]

def add_agent():
    global vehicle_count, destinations, list_origins, list_dests
    
    agent = Vehicle(vehicle_count, random_coord(), 0, 0, 0, random_coord(), 0, 0, 0)
    while agent.coord == agent.dest:
            agent.coord = random_coord()
    agent.steps_od = abs(agent.dest[0] - agent.coord[0]) + abs(agent.dest[1] - agent.coord[1])
    agents.append(agent)
    agent.board_id = board.create_rectangle(agent.coord[0]*Width+Width*2/10, agent.coord[1]*Width+Width*2/10,
                                agent.coord[0]*Width+Width*8/10, agent.coord[1]*Width+Width*8/10, fill="green", width=2)
    agent.t_id = board.create_text(agent.coord[0]*Width+Width*1/2, agent.coord[1]*Width+Width*1/2, text=agent.id)
    agent.board_id_d = board.create_rectangle(agent.dest[0]*Width+Width*2/10, agent.dest[1]*Width+Width*2/10,
                                agent.dest[0]*Width+Width*8/10, agent.dest[1]*Width+Width*8/10, fill="yellow", width=1)
    agent.board_id_d_t = board.create_text(agent.dest[0]*Width+Width*1/2, agent.dest[1]*Width+Width*1/2, text=agent.id)
    vehicle_count += 1 
    list_origins += 1
    if list_origins >= len(origins):
        list_origins = 0
    list_dests += 1
    if list_dests >= len(destinations):
        list_dests = 0

for z in range(starting_agents):
    add_agent()

def can_move(agent, action):
    dx = 0
    dy = 0
    if action == 0: #left
        dx = -1
    elif action == 1: #up
        dy = -1
    elif action == 2: #right
        dx = 1
    elif action == 3: #down
        dy = 1
    new_x = agent.coord[0] + dx
    new_y = agent.coord[1] + dy
    if (new_x >= 0) and (new_x < x) and (new_y >= 0) and (new_y < y) and not ((new_x, new_y) in walls):
        return True
    else:
        return False

def move(agent, action):
    if action == 0: #left
        agent.coord[0] += -1
    elif action == 1: #up
        agent.coord[1] += -1
    elif action == 2: #right
        agent.coord[0] += 1
    elif action == 3: #down
        agent.coord[1] += 1
    board.coords(agent.board_id, agent.coord[0]*Width+Width*2/10, agent.coord[1]*Width+Width*2/10, agent.coord[0]*Width+Width*8/10, agent.coord[1]*Width+Width*8/10)
    board.coords(agent.t_id, agent.coord[0]*Width+Width*5/10, agent.coord[1]*Width+Width*5/10)

def coord_to_index(coord):
    index = coord[1] * x + coord[0]
    return index


def has_arrived(agent): 
    if agent.coord == agent.dest:
        agent.score += -agent.steps_od +1
        scores.append(agent.score)
        return True
    return False

def delete_agent(agent):
    board.delete(agent.board_id)
    board.delete(agent.t_id)
    board.delete(agent.board_id_d)
    board.delete(agent.board_id_d_t)
    
board.grid(row=0, column=0)

def start_simulation():
    master.mainloop()

def run():
    global agents, x, y
    global DQN
    time.sleep(1)
    t = 1
    step = 0
    once = False

    s=np.zeros((1, x*y))
    while True:
        # ---------------------------- Get the state ----------------------------
        state_grid = np.zeros((1, x*y))
        for agent in agents:
            index = coord_to_index(agent.coord)
            state_grid[0, index] += 1
        
        state_grid_previous = state_grid.copy()

        agents_pd = []
        for agent in agents:
            agents_pd.append(agent.coord)
            agents_pd.append(agent.dest)
            agent.reward = 0

        # -------------------------   Select an action   -------------------------
        # s is the state
        # a is the action
        for agent in agents:
            s = np.concatenate((np.array([agent.coord[0],
                                agent.coord[1],
                                agent.dest[0],
                                agent.dest[1]]),
                                state_grid_previous[0]),axis=0)
            a = DQN.choose_action(s)
            agent.action = a
        # ------------------ Move the agent towards the destination ------------------
        for agent in agents:
            #axis = random.randint(0, 1) # X or Y uncomment this for random choosing
            axis = agent.action
            if agent.dest[axis] == agent.coord[axis]: #make sure they are not in the axis already
                if axis == 0:
                    axis = 1
                else:
                    axis = 0
                agent.reward += 0.5
                #agent.score += walk_reward    
            if axis == 0: #axis X 
                if agent.dest[axis] < agent.coord[axis]:
                    if can_move(agent, 0):
                        move(agent, 0)
                    else: #if it can't move on X then move on Y
                        if agent.dest[axis] < agent.coord[axis]:
                            move(agent, 1)
                        else:
                            move(agent, 3)
                else:
                    if can_move(agent, 2):
                        move(agent, 2)
                    else: #if it can't move on X then move on Y
                        if agent.dest[axis] < agent.coord[axis]:
                            move(agent, 1)
                        else:
                            move(agent, 3)
            else: #axis Y
                if agent.dest[axis] < agent.coord[axis]:
                    if can_move(agent, 1):    
                        move(agent, 1)
                    else:
                        if agent.dest[axis] < agent.coord[axis]:
                            move(agent, 0)
                        else:
                            move(agent, 2)
                else:
                    if can_move(agent, 3):
                        move(agent, 3)
                    else:
                        if agent.dest[axis] < agent.coord[axis]:
                            if can_move(agent, 0):
                                move(agent, 0)
                            else: #if it can't move on X then move on Y
                                if agent.dest[axis] < agent.coord[axis]:
                                    move(agent, 1)
                                else:
                                    move(agent, 3)
            agent.score += walk_reward
        # ------------------ Check if the agents have arrived ------------------
        agents_arrived = []
        for agent in agents:
            if has_arrived(agent):
                agents_arrived.append(agent)

        # -------------------------- Get the next state S_  -----------------------------------
        state_grid = np.zeros((1, x*y))
        for agent in agents:
            index = coord_to_index(agent.coord)
            state_grid[0, index] += 1
        
        # ------------------ Apply the rewards according to the traffic ------------------     
        density = {}
        for agent in agents:
            s = str(agent.coord)
            if density.get(s) == None:
                density[s] = []
            density[s].append(agent)
        
        for k, v in density.items():
            for agent in v: 
                agent.score += walk_reward
                if len(v) > 1:
                    agent.reward += 10*(len(v)-1)
                    agent.score += 1*len(v) - 1


        # ------------------ Store the transition (S, A, R, S_ ) ------------------ 
        
        for agent in agents:
            for i in range(len(agents_pd)/2):
                s = np.concatenate((np.array([agents_pd[i][0],
                                    agents_pd[i][1],
                                    agents_pd[i+1][0],
                                    agents_pd[i+1][1]]),
                                    state_grid_previous[0]),axis=0)
                S_ = np.concatenate((np.array([agent.coord[0],
                                agent.coord[1],
                                agent.dest[0],
                                agent.dest[1]]),
                                state_grid[0]),axis=0)
                DQN.store_transition(np.array(s),
                                np.array(agent.action),
                                np.array(agent.reward),
                                np.array(S_))
        
        # ------------------ Delete the arrived agents  ------------------
        new_agents = []
        for agent in agents:
            if agent not in agents_arrived:
                new_agents.append(agent)
            else:    
                delete_agent(agent)
        agents = new_agents
        for z in range(len(agents_arrived)):
            add_agent()

        if step == 401:
            scores = []
        
        if (step > 800) and (step % 5 == 0):
            DQN.learn()
           
        t += 1.0
        step += 1
        
        # MODIFY THIS SLEEP IF THE GAME IS GOING TOO FAST.
        time.sleep(.0000001)
        
        """ To stop the game at step number 
        if step > 10000:
            if once == False:
                once = True
                print "break out of the while loop"
                print DQN.epsilon
                print DQN.learn_step_counter
            break
            time.sleep(3.0)
        """    
    
global DQN
DQN = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.03,
                      reward_decay=0.9,
                      replace_target_iter=150,
                      memory_size=1000,
                      # output_graph=True
                      )
t = threading.Thread(target=run)
t.daemon = True
t.start()
t.join
start_simulation()
DQN.plot_q_t()
DQN.plot_cost()

plot_values = []
accumulation = 0
for i in range(len(scores)):
    accumulation += scores[i]
    if (i+1) % 500 == 0:
        plot_values.append(accumulation/500.0)
        accumulation = 0

print (DQN.epsilon)
print (DQN.learn_step_counter)

#print scores
print (plot_values)
average = np.array(plot_values)
print (np.average(average))
print ("start plot")
plt.plot(plot_values)
plt.ylabel('agents encountered')
plt.show()
