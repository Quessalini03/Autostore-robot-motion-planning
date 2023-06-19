import numpy as np
UP,DOWN,LEFT,RIGHT = 0,1,2,3
class Robot:
    def __init__(self,robotid, pos=None):
        self.robotid = robotid
        # self.boardid = boardid
        self.state = "ALIVE"
        self.pos = pos
        self.old_pos = None
        self.goal = None
        self.shortestpath = None
        self.action = -1
        self.reward = 0
        self.currSight = np.zeros((28),dtype=np.float32)
        self.prevSight = np.zeros((28),dtype=np.float32)
    def move(self,direction):
        self.old_pos = self.pos
        (x, y) = self.pos
        if direction == UP:
            self.pos = (x,y-1)
        elif direction == DOWN:
            self.pos = (x,y+1)
        elif direction == LEFT:
            self.pos = (x-1,y)
        elif direction == RIGHT:
            self.pos =(x+1,y)
    def set_pos(self,pos):
        self.old_pos = self.pos
        self.pos = pos 
    def set_goal(self,goal):
        self.goal = goal
        self.reward = 0
        self.shortestpath = abs(self.pos[0]-self.pos[0])+abs(self.pos[1]-self.goal[1])
    def reach_goal(self):
        return self.pos == self.goal
    def set_sight(self,board):
        board_size = np.shape(board)[0]
        self.prevSight = self.currSight
        self.currSight = np.zeros(28,dtype=np.float32)
        (x, y) = self.pos
        count = 0
        ###UP
        if self.goal[0] == x and y>self.goal[1]: self.currSight[count]=1
        row = y-1
        while row >= 0:
            if board[row][x] == 1:
                self.currSight[count+1] = 1/(abs(y-row))
                break
            row-=1
        self.currSight[count+2] = 1/(y+1)
        count+=3
        ###DOWN
        if self.goal[0] == x and y<self.goal[1]: self.currSight[count]=1
        row = y+1
        while row < board_size:
            if board[row][x] == 1:
                self.currSight[count+1] = 1/(abs(y-row))
                break
            row+=1
        self.currSight[count+2] = 1/(board_size - y)
        count+=3
        ###LEFT
        if self.goal[1] == y and x<self.goal[0]: self.currSight[count]=1
        col = x-1
        while col >=0:
            if board[y][col] == 1:
                self.currSight[count+1] = 1/(abs(x-col))
                break
            col-=1
        self.currSight[count+2] =1/(x+1)
        count+=3
        ###RIGHT
        if self.goal[1] == y and x>self.goal[0]: self.currSight[count]=1
        col = x+1
        while col < board_size:
            if board[y][col] == 1:
                self.currSight[count+1] = 1/(abs(x-col))
                break
            col+=1
        self.currSight[count+2] = 1/(board_size - x) 
        count+=3
        ###UP LEFT
        if self.goal[0] - x == self.goal[1] - y and self.goal[0] < x:
            self.currSight[count]=1
        col = x-1
        row = y-1
        while col>=0 and row>=0:
            if board[row, col] == 1:
                self.currSight[count+1] = 1/(abs(x-col)+abs(y-row))
                break
            col-=1
            row-=1
        self.currSight[count+2] =1/(2*min(x+1,y+1))
        count+=3
        ###UP RIGHT
        if self.goal[0] - x == -1*(self.goal[1] - y) and self.goal[0]>x:
            self.currSight[count]=1
        col = x + 1
        row = y - 1
        while col<board_size and row>=0:
            if board[row, col] == 1:
                self.currSight[count+1]=1/(abs(x-col)+abs(y-row))
                break
            col+=1
            row-=1
        self.currSight[count+2] = 1/(2*min(board_size-x,y+1))
        count+=3
        ###DOWN RIGHT
        if self.goal[0] - x == self.goal[1] - y and self.goal[0]>x:
            self.currSight[count]=1
        col = x+1
        row = y+1
        while row <board_size and col <board_size:
            if board[row,col] == 1:
                self.currSight[count+1]=1/(abs(x-col)+abs(y-row))
                break
            row+=1
            col+=1
        self.currSight[count+2] = 1/(2*min(board_size-x,board_size-y))
        count+=3
        ###DOWN LEFT
        if self.goal[0] - x == -1*(self.goal[1] - y) and self.goal[0]<x:
            self.currSight[count]=1
        col = x-1
        row = y+1
        while row <board_size and col >=0:
            if board[row,col] == 1:
                self.currSight[count+1]=1/(abs(x-col)+abs(y-row))
                break
            row+=1
            col-=1
        self.currSight[count+2] = 1/(2*min(x+1,board_size-y))
        count+=3

        if self.goal[0]<x:
            self.currSight[count]=1
        elif self.goal[0]>x:
            self.currSight[count+1]=1

        if self.goal[1]<y:
            self.currSight[count+2]=1
        elif self.goal[1]>y:
            self.currSight[count+3]=1

