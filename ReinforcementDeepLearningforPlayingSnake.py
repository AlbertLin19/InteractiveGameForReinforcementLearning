# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:46:28 2018

@author: Albert Lin

made a basic version of the classic game, "Snake"
and trained an AI to play it through reinforcement deep learning
"""

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
    def __init__(self, boardSize=20, startingSize=4):
        self.boardSize=boardSize
        self.size = startingSize
        self.numStateInputs = ((20, 20, 1), (2, ), (2, ))
        self.numActions = 3
        self.reset()
    
    def reset(self):
        self.positionList = np.zeros((self.size, 2))
        #filling the position list with starting positions
        self.positionList[0:self.size, 1] = np.arange(self.size)
        self.positionList = self.positionList.astype('intp')
        self.newApplePos()
        self.paintBoard()
        self.score = 0
        self.tick = 0
        self.velocity = 1
        self.done = False
        return self.getStateInput()
    
    def displayInfo(self):
        #plt.imshow(self.board, cmap='gray')
        print(self.board)
        print(self.positionList)
        print(self.applePos)
        print('velocity: ' + str(self.velocity))
        
    def getStateInput(self):
        if self.velocity == 0:
            velVector = np.asarray((-1, 0))
        elif self.velocity == 1:
            velVector = np.asarray((0, 1))
        elif self.velocity == 2:
            velVector = np.asarray((1, 0))
        else:
            velVector = np.asarray((0, -1))
        return [np.reshape(self.board.astype('float32'), (1, ).__add__(self.numStateInputs[0])), np.reshape(velVector.astype('float32'), (1, ).__add__(self.numStateInputs[1])), np.reshape(self.applePos.astype('float32'), (1, ).__add__(self.numStateInputs[2]))]
    
    def takeAction(self, action):
        '''
        0 is turn counterclockwise, 1 is straight, 2 is turn clockwise
        '''
        # translating the input into a change to the velocity
        self.tick+=1
        if action == 0:
            self.velocity -=1
            if self.velocity < 0:
                self.velocity = 3
        elif action == 2:
            self.velocity+=1
            if self.velocity > 3:
                self.velocity = 0
                
        # translating the velocity label to a vector
        # and adding the new position to the end
        if self.velocity == 0:
            velVector = np.asarray((-1, 0))
        elif self.velocity == 1:
            velVector = np.asarray((0, 1))
        elif self.velocity == 2:
            velVector = np.asarray((1, 0))
        else:
            velVector = np.asarray((0, -1))
        self.positionList = np.vstack((self.positionList, (self.positionList[-1]+velVector)))
        
        # checking if an apple was eaten
        # adjusting size and selecting a new apple position if needed
        if self.ateApple():
                self.score+=1
                self.size+=1
                self.newApplePos()
        
        # trimming the position list if too big
        if len(self.positionList) > self.size:
                self.positionList = self.positionList[1:]
        
        # checking if collision occured, and will repaint board if not
        if not self.done:
            self.done = self.collision()
        if not self.done:
            self.paintBoard()
            
        # printing the results of this action
        print("Current Score: {}".format(self.score))
        print("Current Tick: {}".format(self.tick))
        if self.done:
            print('Game Over!')
        
        # returning useful info for the AI input
        reward = self.score*self.boardSize*5-self.snakeDistToApple()*10
        return self.getStateInput(), reward, self.done, None
        
    def collision(self):
        '''
        check to see whether the current position
        is in collision with the board's border or with the snake body
        '''
        positionList = self.positionList
        currentPos = positionList[-1]
        if any(np.array_equal(pos, currentPos) for pos in positionList[:-1])\
        or any(currentPos[:] < 0) or any(self.boardSize <= currentPos[:]):
            print('collision detected')
            return True
        else:
            return False
        
    def ateApple(self):
        '''
        check to see whether an apple was eaten
        '''
        if (np.array_equal(self.positionList[-1], self.applePos)):
            print('ate an apple')
            return True
        
    def newApplePos(self):
        '''
        randomly choose an unoccupied spot to be the next apple position
        '''
        while True:
            randPos = np.asarray((random.randint(0, self.boardSize-1), random.randint(0, self.boardSize-1)), dtype='intp')
            if not any(np.array_equal(pos, randPos) for pos in self.positionList[:]):
                self.applePos = randPos
                print('new apple location selected')
                break
            
    def paintBoard(self):
        '''
        mapping the locations
        in positionList and applePos
        to the board
        '''
        self.board = np.zeros((self.boardSize, self.boardSize))
        self.board[self.positionList[:, 0], self.positionList[:, 1]] = 1
        self.board[self.applePos[0], self.applePos[1]] = -1
        
    def snakeDistToApple(self):
        '''
        calculate distance between apple and snake
        '''
        a = self.positionList[-1][0]-self.applePos[0]
        b = self.positionList[-1][1]-self.applePos[1]
        return np.sqrt(a**2+b**2)

#%%
'''

testing the snake program with user input


env = Snake(boardSize=20, startingSize=4);
env.displayInfo()
while True:
    userInput = float(input())
    print('inputted is: ' + str(userInput))
    if (userInput != 0 and userInput != 1 and userInput !=2):
        print('game exited')
        break
    else:
        print('inputting next step')
        env.takeAction(userInput)
        env.displayInfo()
'''
#%%
'''
This AI model for Deep-Q Learning comes originally from
this github repo:
    https://github.com/keon/deep-q-learning
    
changed the code accordingly to better suit this problem
defining the AI training model
'''
from keras import layers, optimizers, Input, Model
from collections import deque

class AIPlayer:
    
    # initializing constants and constructing model
    def __init__(self, numStateInputs, numActions):
        self.numStateInputs = numStateInputs
        self.numActions = numActions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.getModel()

    def getModel(self):
        # Neural Net for Deep-Q learning Model
        boardInput = Input(shape=self.numStateInputs[0])
        boardOutput = layers.Conv2D(64, (3, 3), activation='relu')(boardInput)
        boardOutput = layers.MaxPooling2D((2, 2))(boardOutput)
        boardOutput = layers.Conv2D(128, (3, 3), activation='relu')(boardOutput)
        boardOutput = layers.MaxPooling2D((2, 2))(boardOutput)
        boardOutput = layers.Conv2D(128, (3, 3), activation='relu')(boardOutput)
        boardOutput = layers.Flatten()(boardOutput)
        velInput = Input(shape=self.numStateInputs[1])
        velOutput = layers.Dense(32, activation='relu')(velInput)
        appleInput = Input(shape=self.numStateInputs[2])
        appleOutput = layers.Dense(32, activation='relu')(appleInput)
        concatenated = layers.concatenate([boardOutput, velOutput, appleOutput], axis=-1)
        actionOutput = layers.Dense(32, activation='relu')(concatenated)
        actionOutput = layers.Dense(self.numActions, activation='softmax')(actionOutput)
        model = Model([boardInput, velInput, appleInput], actionOutput)
        model.compile(loss='mse',
                      optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.numActions)
        actionValues = self.model.predict(state)
        return np.argmax(actionValues[0])  # returns action

    def replay(self, batchSize):
        minibatch = random.sample(self.memory, batchSize)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

#%%
'''
running the AI to train
'''
NumTrainGames = 1000

env = Snake(boardSize=20)
numStateInputs = env.numStateInputs
numActions = env.numActions
player = AIPlayer(numStateInputs, numActions)
done = False
batch_size = 32
highestScore = 0

for game in range(NumTrainGames):
    finalScore = 0
    state = env.reset() # get initial state
    env.displayInfo()
    done = False
    while not done:
        action = player.act(state) # get the action the AI wants to do
        nextState, reward, done, _ = env.takeAction(action) # collect the results from taking the action
        env.displayInfo()
        reward = reward if not done else -2000 # keep reward unless game ended
        player.remember(state, action, reward, nextState, done)
        state = nextState
        if len(player.memory) > batch_size:
            player.replay(batch_size)
    score = env.score
    print("Game: {}/{}, score: {}, epsilon: {:.2}".format(game, NumTrainGames, score, player.epsilon))
    finalScore = score
        
    if finalScore > highestScore:
        # player.save("/Users/Albert Lin/Documents/GitHub/score{}".format(finalScore)) is this path string correct?
        print('NEW HIGH SCORE: {}!'.format(finalScore))
        print('implement model saving!!!')
        highestScore = finalScore