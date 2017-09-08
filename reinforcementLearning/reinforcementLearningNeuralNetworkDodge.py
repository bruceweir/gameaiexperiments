# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:01:42 2017

@author: brucew
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:51:15 2017

@author: brucew

A simple game showing how a neural network can learn q_values (state/action choices)

"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

import random
import time

actions = ['left', 'right', 'wait']
width = 5
height = 10 


def learn_to_play():

    #Initially, the agent begins at state 0, the far left of the environment
    agent_position = random.choice(range(width))
    environment = update_environment(agent_position)
    #He will move entirely randomly at first
    epsilon_greedy = 1.0
    #Each step in a single game is recorded in the game_memory
    game_memory = []
    #The replay memory holds the steps of every game until a training run is needed
    replay_memory = []
    model = create_neural_network()
    n_games = 0
    n_training_runs = 0
    run = True
        
    while run:
        #pick an action to perform
        action, model_q_predictions = choose_action(environment, model, epsilon_greedy)
        previous_environment = np.copy(environment)
        #update the state of the environment, and return a score and (if the agent has left the right hand edge of 
        #the environment) a termination trigger        
        environment, agent_position, terminate_game, score = perform_action(environment, agent_position, action)        
        #record the move that was taken
        store_action_in_game_memory(previous_environment, action, model_q_predictions, game_memory)
        
           
        if terminate_game:
                        
            #go backwards through the game history and update the q_values to 
            #reflect the score received
            update_q_values_for_this_game(score, game_memory)
            replay_memory.append(game_memory[:])
            game_memory = []
            agent_position = random.choice(range(width))
            n_games += 1
            
            if n_games == 10:
                #once enough games have been played, train the network with the 
                #game states as input data, and the measured q_values as the target
                train_network(model, replay_memory)
                replay_memory = []
                n_games = 0
                n_training_runs += 1
                epsilon_greedy *= 0.9
                
                
        

        if n_training_runs == 20:
            run=False
    
    return model
            
 #       time.sleep(.1)

def update_environment(agent_position):
    
    environment = np.zeros((height, width))
    environment[height-1][agent_position] = 1
    return environment


def create_neural_network():
    
    model = Sequential()
    model.add(Dense(height*width, input_shape=(height*width,)))
    model.add(Dense(len(actions), activation='elu')) #there are 2 actions
    model.compile(optimizer='adam',
          loss='mean_squared_error')
    
    return model
    
#Either choose the action that the model predicts is best, or a
#random action, depending upon the value of epsilon_greedy
def choose_action(environment, model, epsilon_greedy=0.0):
        
    action = 0
    model_prediction = list(model.predict(np.array([environment.reshape(width*height)]))[0])
    
    if random.random() > epsilon_greedy:
        action = np.argmax(model_prediction)
    else:
        action = random.choice(range(len(actions)))
    
    return action, model_prediction

#update the state of the environment, depending upon the action taken by 
#the agent
def perform_action(environment, agent_position, action):
    
    termination = False
    score = 0
    
    if action == 0 and agent_position > 0: #left
        agent_position -=1
    elif action == 1 and agent_position < width-1: #right
        agent_position += 1
    
    if action == 1 and agent_position == width-1: #termination condition
        termination = True
        score = 1
    
    environment = update_environment(agent_position)
    return environment, agent_position, termination, score

"""    
Remember the state for the current step in the game, the action that was taken
from here, and the prediction that the model made for the q_values (action choices)
(if the agent moved at random)    
"""
def store_action_in_game_memory(environment, action, predicted_q_values, game_memory):
    state_result = {}
    state_result['state'] = list(np.copy(environment.reshape(width*height)))
    state_result['action'] = action
    state_result['Q_values'] = list(predicted_q_values[:])
    game_memory.append(state_result)
 
    
"""Go backwards through the game history, updating the q_value for the action
that was made according to the utility that was gained by this action
i.e. For the action that scored a point, the q_value is the value of
that point. For the step just before this, the q_value is the value of 
the point multiplied by the discount_rate, and so on back to the first move
of the game
"""
def update_q_values_for_this_game(score, game_memory):

    discount_rate = 0.97
    
    for x in reversed(range(len(game_memory))):
        action = game_memory[x]['action']
        game_memory[x]['Q_values'][action] = score
        score *= discount_rate


def train_network(model, replay_memory):
    x_train, y_train, x_test, y_test = create_training_data(replay_memory)

    model.fit(x_train, y_train,
              batch_size=16,
              epochs=100,
              verbose=True,
              validation_data=(x_test, y_test))    
    

"""
The training inputs are the game state and target outputs are the
q_values which we have been measuring as we have been playing the game
"""
def create_training_data(replay_memory):
    
    print(replay_memory)
    x_data=[]
    y_data=[]
    for game in replay_memory:
        x_data.extend([gamestep['state'] for gamestep in game])
        y_data.extend([gamestep['Q_values'] for gamestep in game])
    
    split_position = int(0.9  * len(y_data))
    
    x_train = np.array(x_data[:split_position])
    x_test = np.array(x_data[split_position:])
    y_train = np.array(y_data[:split_position])
    y_test = np.array(y_data[split_position:])
    
    return x_train, y_train, x_test, y_test


def play_game_using_model(model, initial_position=0):
    
    plt.figure()
    plt.ion()
    
    terminate_game = False
    agent_position = initial_position
    environment = update_environment(agent_position)
    
    while not terminate_game:
        draw_environment(environment)
        #print(environment)
        action, _ = choose_action(environment, model)
        environment, agent_position, terminate_game, _ = perform_action(environment, agent_position, action)        
        time.sleep(1)


def draw_environment(environment):
    
    plt.imshow(environment, cmap='gray')
    plt.pause(0.01)
    plt.show()
    
    print(environment, end='\r\r')
    
        
    #draw_environment(environment)
    
#main()
