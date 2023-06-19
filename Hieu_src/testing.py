import torch
import numpy as np
import pygame as pg
from NeuralNetwork import NeuralNetwork
from Robot import Robot
BASEX = 100
BASEY = 100
CELLWIDTH = 40
UP,DOWN,LEFT,RIGHT = 0,1,2,3
class Visualize:
    def __init__(self,numRobots,storeSize):
        pg.init()
        self.storeSize = storeSize
        self.screen = pg.display.set_mode((600,600))
        self.clock = pg.time.Clock()
        self.grid = np.zeros((storeSize,storeSize),dtype=int)
        self.robots = []
        self.goal_pos = []
        for i in range(numRobots):
            self.robots.append(Robot(i))
        x = np.random.randint(0,storeSize)
        y = np.random.randint(0,storeSize)
        for i in range(numRobots):
            while self.grid[y,x] == 1:
                x = np.random.randint(0,storeSize)
                y = np.random.randint(0,storeSize)
            self.grid[y,x]=1
            self.robots[i].set_pos((x,y))
        self.model = NeuralNetwork(28,4)
        self.model.load("curr_model.pth")
        self.model.eval()
    def random_goal(self,pos):
        x = np.random.randint(0,self.storeSize)
        y = np.random.randint(0,self.storeSize)
        while (x,y) in self.goal_pos or (x,y)==pos:
            x = np.random.randint(0,self.storeSize)
            y = np.random.randint(0,self.storeSize)
        return (x,y)
    def preset(self):
        for robot in self.robots:
            goal = self.random_goal(robot.pos)
            self.goal_pos.append(goal)
            robot.set_goal(goal)
            robot.set_sight(self.grid)
    def background(self):
        self.screen.fill("white")
    def board(self):
        for row in range(self.storeSize+1):
            pg.draw.line(self.screen,"black",(BASEX,BASEY+row*CELLWIDTH),
                         (BASEX+self.storeSize*CELLWIDTH,BASEY+row*CELLWIDTH))
        for col in range(self.storeSize+1):
            pg.draw.line(self.screen,"black",(BASEX+col*CELLWIDTH,BASEY),
                         (BASEX+col*CELLWIDTH,BASEY+self.storeSize*CELLWIDTH))
    def drawRobot(self):
        for robot in self.robots:
            (x,y) = robot.pos
            points = [
                (BASEX+x*CELLWIDTH,BASEY+y*CELLWIDTH),
                (BASEX+(x+1)*CELLWIDTH,BASEY+y*CELLWIDTH),
                (BASEX+(x+1)*CELLWIDTH,BASEY+(y+1)*CELLWIDTH),
                (BASEX+x*CELLWIDTH,BASEY+(y+1)*CELLWIDTH)
            ]
            pg.draw.polygon(self.screen,"yellow",points)
            font = pg.font.Font(None, 30)
            text = font.render(f"{robot.robotid}",True,"black")
            text_rect = text.get_rect()
            text_rect.center = ((points[0][0]+points[2][0])/2,(points[0][1]+points[2][1])/2)
            self.screen.blit(text,text_rect)
            (x,y) = robot.goal
            points = [
                (BASEX+x*CELLWIDTH,BASEY+y*CELLWIDTH),
                (BASEX+(x+1)*CELLWIDTH,BASEY+y*CELLWIDTH),
                (BASEX+(x+1)*CELLWIDTH,BASEY+(y+1)*CELLWIDTH),
                (BASEX+x*CELLWIDTH,BASEY+(y+1)*CELLWIDTH)
            ]
            pg.draw.polygon(self.screen,"green",points)
            font = pg.font.Font(None, 30)
            text = font.render(f"{robot.robotid}",True,"black")
            text_rect = text.get_rect()
            text_rect.center = ((points[0][0]+points[2][0])/2,(points[0][1]+points[2][1])/2)
            self.screen.blit(text,text_rect)
    def draw_dead(self,pos):
        (x,y) = pos
        points = [
            (BASEX+x*CELLWIDTH,BASEY+y*CELLWIDTH),
            (BASEX+(x+1)*CELLWIDTH,BASEY+y*CELLWIDTH),
            (BASEX+(x+1)*CELLWIDTH,BASEY+(y+1)*CELLWIDTH),
            (BASEX+x*CELLWIDTH,BASEY+(y+1)*CELLWIDTH)
        ]
        pg.draw.polygon(self.screen,"red",points)
    def isValidMove(self,pos,direction,state):
        (x,y) = pos
        if direction == UP: y-=1
        elif direction == DOWN: y+=1
        elif direction ==  LEFT: x-=1
        elif direction == RIGHT: x+=1
        if x==-1 or y==-1 or x==self.storeSize or y==self.storeSize \
            or state[y, x]==1:
            return False
        return True
    def make_action(self,curr_grid):
        for i in range(len(self.robots)):
            self.robots[i].set_sight(curr_grid)
            input_tensor = torch.tensor(np.reshape(self.robots[i].currSight,(-1,28)))
            q_values = self.model(input_tensor)
            with open(f"log_robot_{i}.txt",mode='a',encoding='utf-8') as f:
                f.write(str(q_values))
                f.write('\n')
                f.write(str(input_tensor))
            action = torch.argmin(q_values,1).detach()
            self.robots[i].action = action
    def info(self,t,dead,success,colision):
        font = pg.font.Font(None, 30)
        text = font.render(f"t: {t}",True,"black")
        text_rect = text.get_rect()
        text_rect.center = (50,20)
        self.screen.blit(text,text_rect)
        font = pg.font.Font(None, 30)
        text = font.render(f"Dead: {dead}",True,"black")
        text_rect = text.get_rect()
        text_rect.center = (50,40)
        self.screen.blit(text,text_rect)
        font = pg.font.Font(None, 30)
        text = font.render(f"Duccess: {success}",True,"black")
        text_rect = text.get_rect()
        text_rect.center = (50,60)
        self.screen.blit(text,text_rect)
        font = pg.font.Font(None, 30)
        text = font.render(f"Colision: {colision}",True,"black")
        text_rect = text.get_rect()
        text_rect.center = (50,80)
        self.screen.blit(text,text_rect)
    def run(self,num):
        self.preset()
        t = 0
        dead = 0
        success=0
        colision = 0
        while dead+success <= num:
            self.background()
            self.board()
            self.drawRobot()
            curr_grid = np.copy(self.grid)
            self.make_action(curr_grid)
            self.info(t,dead,success,colision)
            pg.display.flip()
            pg.time.delay(1000)
            pos_colision = {}
            for robot in self.robots:
                if not self.isValidMove(robot.pos,robot.action,curr_grid):
                    dead+=1
                    self.draw_dead(robot.pos)
                    pg.display.flip()
                    pg.time.delay(1000)
                    self.goal_pos.remove(robot.goal)
                    goal = self.random_goal(robot.pos)
                    robot.set_goal(goal)
                    self.goal_pos.append(goal)
                else: 
                    robot.move(robot.action)
                    (x,y)=robot.pos
                    (oldx,oldy)=robot.old_pos
                    self.grid[y,x] = 1
                    self.grid[oldy, oldx] = 0
                    if (x,y) not in pos_colision.keys():
                        pos_colision[(x,y)]=1
                    else: pos_colision[(x,y)]+=1
                if robot.reach_goal():
                    success+=1
                    self.goal_pos.remove(robot.goal)
                    goal = self.random_goal(robot.pos)
                    robot.set_goal(goal)
                    self.goal_pos.append(goal)
            for v in pos_colision.values():
                if v>1: colision+=1
            t+=1

visual = Visualize(4,10)
visual.run(10)