# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:46:28 2018

@author: Albert Lin

made a basic version of the classic game, "Snake"
and trained an AI to play it through reinforcement deep learning
"""
#%%
import numpy as np
import random

print("Which mode to run this in? (T)rain/(U)serplay/(M)odelplay/(W)atch")
mode = input()

class Snake():
    '''
    programming the classic snake game in a simple manner
    you start as a one-unit length snake
    each apple you eat increases your length and your score
    if you move into yourself or a wall, you lose
    the objective is to recieve the highest score possible
    
    '''
    # direction definition:
    # 0 - up, 1 - right, 2 - down, 3 - left
    
    #!!! CLASS VARIABLES
    #board, boardSize
    #positionList, applePos, velocity
    #score, tick
    
    #!!! NOTE:
    # apple position and snake positions are kept in lists
    # the state of the game is kept in the board array, which will
    # be updated with the apple and snake positions with a call to paintBoard()
    
    #!!! NOTE:
    # when dealing with multi-dim arrays, use [a, b]
    # NOT [a][b], which is less efficient and can cause confusing indexing problems
    def __init__(self, boardSize=20, startingSize=4):
        '''
        configure a new game environment, with board size and starting size
        initializes a few fields and calls a reset to the game
        '''
        self.boardSize=boardSize
        self.maxDist = (boardSize-1)*2
        self.startingSize = startingSize
        self.numStateInputs = ((boardSize, boardSize, 3), )
        self.numActions = 4
        self.minSnakeColor = 0.5
        self.reset()
    
    def reset(self):
        '''
        resets size, positionList, and apple pos to new state
        paints the board
        sets tick and score to 1 and 0 respectively
        make an array for body gradient
        returns initial state
        '''
        self.size = self.startingSize
        self.positionList = np.zeros((self.size, 2))
        #filling the position list with starting positions
        self.positionList[0:self.size, 1] = np.arange(self.size)
        self.positionList = self.positionList.astype('intp')
        self.newApplePos()
        self.bodyGradient = np.linspace(self.minSnakeColor, 1, num=self.startingSize)
        self.paintBoard()
        self.score = 0
        self.tick = 1
        self.done = False
        return self.getStateInput()
        
    def getStateInput(self):
        '''
        returns game board
        IMPORTANT NOTE:
            need to supply inputs as (samples, dim, dim, ..., features)
            as ONE array, not a tuple of arrays
        '''
        return np.reshape(self.board, (1, ).__add__(self.numStateInputs[0]))
    
    def takeAction(self, action):
        '''
        
        inputs an action
        0 - up, 1 - right, 2 - down, 3 - left
        
        updates the tick
        adds new position
        adds to score if apple eaten
        trims snake down to right size
        checks collision
        repaints board if no collision
        
        returns current state, an appropriate reward, and whether game over
        
        if action tries to move backwards, then reward will be negative
        but snake will continue to move forward
        '''
        self.tick+=1
                
        # and adding the new position to the end
        assert action == 0 or action == 1 or action == 2 or action == 3, "Error: only actions 0-3 are allowed!"
        if action == 0:
            velVector = np.asarray((-1, 0))
        elif action == 1:
            velVector = np.asarray((0, 1))
        elif action == 2:
            velVector = np.asarray((1, 0))
        elif action == 3:
            velVector = np.asarray((0, -1))
        self.positionList = np.vstack((self.positionList, (self.positionList[-1]+velVector)))
        
        # check if snake went backwards, correct path if so
        wentBackwards = False
        if np.array_equal(self.positionList[-1], self.positionList[-3]):
            wentBackwards = True
            self.positionList[-1] = self.positionList[-2]*2-self.positionList[-3]
        
        # checking if an apple was eaten
        # adjusting size and selecting a new apple position if needed
        # add to reward if eaten
        reward = 0
        if self.ateApple():
                self.score+=1
                self.size+=1
                self.newApplePos()
                reward+=200
        
        # trimming the position list if too big
        if len(self.positionList) > self.size:
                self.positionList = self.positionList[1:]
        
        # checking if collision occured, and will repaint board if not
        if not self.done:
            self.done = self.collision()
        
        if not self.done:
            self.paintBoard()
        
        # returning useful info for the AI input
        reward += self.getCurrentReward()
        
        if wentBackwards:
            reward = -10
        if self.done:
            reward = -80
            
        return self.getStateInput(), reward, self.done
        
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
        change apple position to randomly selected one
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
        using a gradient for the snake body to indicate body path
        updates the gradient if necessary
        
        apple is red, snake is green, head is blue
        '''
        self.board = np.zeros((self.boardSize, self.boardSize, 3), dtype='float32')
        
        # creating a gradient for snake body
        if self.positionList.size/2 != self.bodyGradient.size:
            self.bodyGradient = np.linspace(self.minSnakeColor, 1, num=self.positionList.size/2, dtype='float32')
        self.board[self.positionList[:-1, 0], self.positionList[:-1, 1], 1] = self.bodyGradient[:-1]
        self.board[self.positionList[-1, 0], self.positionList[-1, 1], 2] = 1
        self.board[self.applePos[0], self.applePos[1], 0] = 1
        
    def snakePathDistToApple(self):
        '''
        calculate sum of magnitude of displacement vectors between apple and snake
        (path distance)
        '''
        a = self.positionList[-1][0]-self.applePos[0]
        b = self.positionList[-1][1]-self.applePos[1]
        return abs(a)+abs(b)
    
    def getCurrentReward(self):
        return self.maxDist-self.snakePathDistToApple()
        
    def getGameInfo(self):
        '''
        returns tick and score
        '''
        
        return self.tick, self.score

#%%
'''

testing the snake program with user input if correct mode

NOTE: necessary to use plt.pause() for a crude animation
since it takes time to plot - better method is to use matplotlib's animation tools
or put it on another thread perhaps?
'''

if mode.__eq__("U"):
    print("Playing Snake with user input")
    import matplotlib.pyplot as plt
    playing = True
    while playing:
        print("Starting new game...")
        env = Snake(boardSize=20, startingSize=4);
        state = env.reset()
        tick, score = env.getGameInfo()
        print("Current Tick: {}".format(tick))
        print("Current Score: {}".format(score))
        imgplot = plt.imshow(state[0])
        plt.pause(0.001)
        alive = True
        while alive:
            print("type 0, 1, 2, or 3")
            userInput = float(input())
            print('inputted is: ' + str(userInput))
            if (userInput != 0 and userInput != 1 and userInput !=2 and userInput!=3):
                print('bad input: ' + str(userInput))
            else:
                print('inputting next step')
                state, reward, done = env.takeAction(userInput)
                tick, score = env.getGameInfo()
                print("Current Tick: {}".format(tick))
                print("Current Score: {}".format(score))
                print("Current Reward: {}".format(reward))
                print("Done: {}".format(done))
                imgplot = plt.imshow(state[0])
                plt.pause(0.001)
            if done:
                alive = False
                
        print('game over')
        print('restart? (y)')
        userInput = input()
        if (userInput.__eq__("y")):
            print("playing again")
        else:
            print("quitting")
            playing = False
#%%
'''
This AI model for Deep-Q Learning comes originally from
this github repo:
    https://github.com/keon/deep-q-learning
    
changed the code accordingly to better suit this problem
defining the AI training model
'''

if mode.__eq__("T") or mode.__eq__("M"):
    print("Loading AI model class...")
    from keras import layers, optimizers, models
    from collections import deque
    
    class AIPlayer:
        
        # initializing constants and constructing model
        def __init__(self, numStateInputs, numActions):
            self.numStateInputs = numStateInputs
            self.numActions = numActions
            self.memory = deque(maxlen=2000)
            self.gamma = 0.97    # discount rate for future events
            self.epsilon = 1.0  # exploration rate
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.997
            self.learning_rate = 0.001
            self.model = self.getModel()
    
        def getModel(self):
            # Neural Net for Deep-Q learning Model
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.numStateInputs[0]))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            #now to flatten the list of 3D vectors into a list of 1D vectors for Dense layers
            model.add(layers.Flatten())
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(self.numActions, activation='linear'))
            model.compile(loss='mse',
                          optimizer=optimizers.Adam(lr=self.learning_rate))
            return model
    
        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))
    
        def act(self, state, exploring=True):
            if np.random.rand() <= self.epsilon and exploring:
                print("Exploring!")
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
    
        def load(self, path):
            self.model.load_weights(path)
    
        def save(self, path):
            self.model.save_weights(path)

#%%
'''
let a trained model play with the following functions
'''
if mode.__eq__("M"):
    print("Loading functions to load and play with a pre-trained AI model")
    def playSnake(player):
        gameEnv = Snake(boardSize=20, startingSize=4)
        state = gameEnv.reset() # get initial state
        tick, score = gameEnv.getGameInfo()
        print("Current Tick: {}".format(tick))
        print("Current Score: {}".format(score))
        reward = gameEnv.getCurrentReward()
        print("Current Reward: {}".format(reward))
        done = False
        while not done:
            action = player.act(state, exploring=False) # get the action the AI wants to do
            print("Taking action: {}".format(action))
            nextState, reward, done = gameEnv.takeAction(action) # collect the results from taking the action
            tick, score = gameEnv.getGameInfo()
            print("Current Tick: {}".format(tick))
            print("Current Score: {}".format(score))
            print("Current Reward: {}".format(reward))
            print("___________________________________________")
            state = nextState
        print("Final Score: {}".format(gameEnv.score))
        
    print("Type full path to model save that you wish to load: ")
    path = input()
    print("Loading from: {}".format(path))
    numStateInputs = env.numStateInputs
    numActions = env.numActions
    player = AIPlayer(numStateInputs, numActions)
    player.load(path)
    print("Playing Snake with loaded model...")
    playSnake(player)
    
#%%
'''
running the AI to train
'''
if mode.__eq__("T"):
    import os
    print("How many games to train for? (Default: 20,000)")
    NumTrainGames = int(input())
    print("Starting to train the model with {} games...".format(NumTrainGames))
    
    savingBoardHistory = False
    print("Save board history? (y/n)")
    usrIn = input()
    if usrIn.__eq__("y"):
        savingBoardHistory = True
        print("Save to where? (path without last slash, i.e. ~ or /home/usr)")
        boardSavePath = input()
        print("Saving board history to: {}".format(boardSavePath))
    else:
        print("Not saving board history")
        
    savingModel = False
    print("Save models? (y/n)")
    usrIn = input()
    if usrIn.__eq__("y"):
        savingModel = True
        print("Save to where? (path without last slash, i.e. ~ or /home/usr)")
        modelSavePath = input()
        print("Saving models to: {}".format(modelSavePath))
    else:
        print("Not saving models")
    
    env = Snake(boardSize=20, startingSize=5)
    numStateInputs = env.numStateInputs
    numActions = env.numActions
    player = AIPlayer(numStateInputs, numActions)
    done = False
    batch_size = 64
    highestScore = 0
    
    for game in range(1, NumTrainGames+1):
        state = env.reset() # get initial state
        print("Current High Score: {}".format(highestScore))
        print("Current Game: {}/{}".format(game, NumTrainGames))
        tick, score = env.getGameInfo()
        print("Current Tick: {}".format(tick))
        print("Current Score: {}".format(score))
        print("___________________________________________")
        if savingBoardHistory:
            reward = env.getCurrentReward()
            cboardSavePath = boardSavePath+"/Game{:06d}".format(game)
            os.mkdir(cboardSavePath)
            np.save(cboardSavePath+"/Tick{:05d}Score{:03d}Reward{:03d}".format(tick, score, reward), state)
        done = False
        while not done:
            action = player.act(state) # get the action the AI wants to do
            print("Taking action: {}".format(action))
            nextState, reward, done = env.takeAction(action) # collect the results from taking the action
            print("Current High Score: {}".format(highestScore))
            print("Current Game: {}/{}".format(game, NumTrainGames))
            tick, score = env.getGameInfo()
            print("Current Tick: {}".format(tick))
            print("Current Score: {}".format(score))
            print("Current Reward: {}".format(reward))
            print("Done: {}".format(done))
            print("___________________________________________")
            if savingBoardHistory:
                np.save(cboardSavePath+"/Tick{:05d}Score{:03d}Reward{:03d}".format(tick, score, reward), nextState)
            player.remember(state, action, reward, nextState, done)
            state = nextState
            if len(player.memory) > batch_size:
                print("Training AI model with memory...")
                player.replay(batch_size)
        score = env.score
        print("Finished Game: {}/{}, score: {}, epsilon: {:.2}".format(game, NumTrainGames, score, player.epsilon))
        if savingBoardHistory:
            os.rename(cboardSavePath, cboardSavePath+"Score{:03d}".format(score))
        if score > highestScore:
            highestScore = score
            print('NEW HIGH SCORE: {}!'.format(score))
            if savingModel:
                player.save(modelSavePath+"/Score{:03d}".format(highestScore))
            
#%%
'''
functions to watch a pre-recorded game
'''

if mode.__eq__("W"):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    fig, title = plt.subplots()
    import os
    print("Where to watch games from? (path without last slash, i.e. ~ or /home/usr)")
    gamePath = input()
    #!!! FIRST LOAD ALL DATA INTO ANIMATOR, THEN PLAY ANIMATION
    images = []
    for gameFolder in os.listdir(gamePath):
        for gameTick in os.listdir(gamePath+"\\"+gameFolder):
            print("Loading: {} {}".format(gameFolder, gameTick))
            im = plt.imshow(np.load(gamePath+"\\"+gameFolder+"\\"+gameTick)[0], animated=True)
            text = title.text(0.5, 19, 'G: '+gameFolder[4:10]+' T: '+gameTick[4:9]+' S: '+gameTick[14:17]+' R: '+gameTick[23:-4], size='x-large', va='bottom', ha='left', color='w')
            images.append([im, text])
    ani = animation.ArtistAnimation(fig, images, interval=25, blit=True, repeat=False)
    #ani.save(path)
    plt.show()