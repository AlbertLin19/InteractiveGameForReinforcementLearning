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
    def __init__(self, boardSize=20, startingSize=3):
        self.boardSize=boardSize
        self.size = startingSize
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.boardSize, self.boardSize))
        self.board[0][0] = 1
        self.positionList = np.zeros((1, 2))
        self.newApplePos()
        self.score = 0
        self.tick = 0
        self.velocity = 1
        
    def displayBoard(self):
        plt.imshow(self.board, cmap='gray')
        
    def getInput(self):
        return self.board
    
    def takeAction(self, action):
        '''
        0 is turn counterclockwise, 1 is straight, 2 is turn clockwise
        '''
        velocity = self.velocity
        positionList = self.positionList
        if action == 0:
            velocity -=1
            if velocity < 0:
                velocity = 3
        elif action == 2:
            velocity+=1
            if velocity > 3:
                velocity = 0
        if velocity == 0:
            velVector = np.asarray((0, -1))
        elif velocity == 1:
            velVector = np.asarray((1, 0))
        elif velocity == 2:
            velVector = np.asarray((0, 1))
        else:
            velVector = np.asarray((-1, 0))
        np.append(positionList, (positionList[-1]+velVector))
        done = self.collision()
        if not done:
            if self.ateApple():
                self.score+=1
                self.size+=1
                self.newApplePos()
        if len(positionList) > self.size:
            delPos, positionList = positionList[0], positionList[1:]
            self.board[delPos[0]][delPos[1]] = 0
        
    def collision(self):
        '''
        check to see whether the current position
        is in collision with the board's border or with the snake body
        '''
        positionList = self.positionList
        currentPos = positionList[-1]
        if any(pos==currentPos for pos in positionList[:-1])\
        or currentPos[:] < 0 or self.boardSize < currentPos[:]:
            return True
        else:
            return False
        
    def ateApple(self):
        if self.positionList[-1]==self.applePos:
            return True
        
    def newApplePos(self):
        '''
        randomly choose an unoccupied spot to be the next apple position
        '''
        while True:
            randX = random.randint(0, self.boardSize)
            randY = random.randint(0, self.boardSize)
            if (self.positionList[randX, randY]==0):
                self.applePos = np.asarray((randX, randY))
                break
