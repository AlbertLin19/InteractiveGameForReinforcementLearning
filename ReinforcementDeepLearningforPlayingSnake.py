# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:46:28 2018

@author: Albert Lin

made a basic version of the classic game, "Snake"
and trained an AI to play it through reinforcement deep learning
"""

import matplotlib.pyplot as plt
import numpy as np
import random
#%%
class Snake():
    '''
    programming the classic snake game in a simple manner
    you start as a one-unit length snake
    each apple you eat increases your length and your score
    if you move into yourself or a wall, you lose
    the objective is to recieve the highest score possible
    
    '''
    # velocity definition:
    # 0 - up, 1 - right, 2 - down, 3 - left
    
    #!!! CLASS VARIABLES
    #board, boardSize
    #positionList, applePos, velocity
    #score, tick
    
    #!!! NOTE:
    # when dealing with multi-dim arrays, use [a, b]
    # NOT [a][b], which is less efficient and can cause confusing indexing problems
    def __init__(self, boardSize=20, startingSize=3):
        self.boardSize=boardSize
        self.size = startingSize
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.boardSize, self.boardSize))
        self.board[0, 0:self.size] = 1
        self.positionList = np.zeros((self.size, 2))
        #filling the position list with starting positions
        self.positionList[0:self.size, 1] = np.arange(self.size)
        self.positionList = self.positionList.astype('intp')
        self.newApplePos()
        self.score = 0
        self.tick = 0
        self.velocity = 1
        
    def displayBoard(self):
        #plt.imshow(self.board, cmap='gray')
        print(self.board)
        print(self.positionList)
        print(self.applePos)
        print(self.velocity)
    def getInput(self):
        return self.board
    
    def takeAction(self, action):
        '''
        0 is turn counterclockwise, 1 is straight, 2 is turn clockwise
        '''
        
        self.tick+=1
        if action == 0:
            self.velocity -=1
            if self.velocity < 0:
                self.velocity = 3
        elif action == 2:
            self.velocity+=1
            if self.velocity > 3:
                self.velocity = 0
        if self.velocity == 0:
            velVector = np.asarray((-1, 0))
        elif self.velocity == 1:
            velVector = np.asarray((0, 1))
        elif self.velocity == 2:
            velVector = np.asarray((1, 0))
        else:
            velVector = np.asarray((0, -1))
        self.positionList = np.vstack((self.positionList, (self.positionList[-1]+velVector)))
        done = self.collision()
        if not done:
            if self.ateApple():
                self.score+=1
                self.size+=1
                self.newApplePos()
            if len(self.positionList) > self.size:
                self.board[self.positionList[0, 0], self.positionList[0, 1]] = 0
                self.positionList = self.positionList[1:]
            print("Current Score: {}".format(self.score))
            print("Current Tick: {}".format(self.tick))
        self.board[self.positionList[:, 0], self.positionList[:, 1]] = 1
        return (self.board, self.velocity, self.applePos)
        
    def collision(self):
        '''
        check to see whether the current position
        is in collision with the board's border or with the snake body
        '''
        positionList = self.positionList
        currentPos = positionList[-1]
        if any(np.array_equal(pos, currentPos) for pos in positionList[:-1])\
        or any(currentPos[:] < 0) or self.boardSize < any(currentPos[:]):
            print('collision detected')
            return True
        else:
            return False
        
    def ateApple(self):
        if (np.array_equal(self.positionList[-1], self.applePos)):
            print('ate an apple')
            return True
        
    def newApplePos(self):
        '''
        randomly choose an unoccupied spot to be the next apple position
        '''
        while True:
            randX = random.randint(0, self.boardSize-1)
            randY = random.randint(0, self.boardSize-1)
            if (self.board[randX, randY]==0):
                self.applePos = np.asarray((randX, randY))
                self.board[randX, randY]=2
                print('new apple created')
                break

#%%
'''
testing the snake program with user input
'''

env = Snake(boardSize=20, startingSize=3);
env.displayBoard()
while True:
    userInput = float(input())
    print('inputted is: ' + str(userInput))
    if (userInput != 0 and userInput != 1 and userInput !=2):
        print('game exited')
        break
    else:
        print('inputting next step')
        env.takeAction(userInput)
        env.displayBoard()
        
        