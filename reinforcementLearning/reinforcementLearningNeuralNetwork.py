# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:51:15 2017

@author: brucew

A simple game showing how a neural network can learn q_values (state/action choices)

The aim of the game is for the agent to move off the furthest right-hand
end of the environment. When he does so, he will receive 1 point

There are two actions that an agent can take for each game step: move 1
step left, or move 1 step right

At first, the agent knows nothing about the environment or even what he
is supposed to do. He learns the results of his actions by exploring his environmet

"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
import time

actions = ['left', 'right']
states = list(range(5))
    
def learn_to_play():

    #Initially, the agent begins at state 0, the far left of the environment
    current_state = 0 
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
        action, model_q_predictions = choose_action(current_state, model, epsilon_greedy)
        previous_state = current_state
        #update the state of the environment, and return a score and termination trigger        
        current_state, terminate, score = perform_action(current_state, action)        
        #record the move that was taken
        store_action_in_game_memory(previous_state, action, model_q_predictions, game_memory)
           
        if terminate:
            
            current_state = 0
            #go backwards through the game history and update the q_values to 
            #reflect the score received
            update_q_values_for_this_game(score, game_memory)
            replay_memory.append(game_memory[:])
            game_memory = []
            n_games += 1
            
            if n_games == 10:
                #once enough games have been played, train the network with the 
                #game states as input data, and the measured q_values as the target
                train_network(model, replay_memory)
                replay_memory = []
                n_games = 0
                n_training_runs += 1
                epsilon_greedy *= 0.9
                
                
        draw_environment(current_state)

        if n_training_runs == 20:
            run=False
    
    return model
            
 #       time.sleep(.1)

def draw_environment(currentState):
    
    world = ''    
    for _ in range(0, currentState):
        world += ' _'    
    world = world + ' X'    
    for _ in range(currentState+1, len(states)):
        world += ' _'
    
    print(world, end='\r', flush=True)

def create_neural_network():
    
    model = Sequential()
    model.add(Dense(10, input_shape=(5,)))#the environment is 5 units long
    model.add(Dense(2, activation='elu')) #there are 2 actions
    model.compile(optimizer='adam',
          loss='mean_squared_error')
    
    return model
    
#Either choose the action that the model predicts is best, or a
#random action, depending upon the value of epsilon_greedy
def choose_action(currentState, model, epsilon_greedy=0.0):
    
    input_vector = create_input_vector(currentState)
    action = 0
    model_prediction = list(model.predict(np.array([input_vector]))[0])
    
    if random.random() > epsilon_greedy:
        action = np.argmax(model_prediction)
    else:
        action = random.choice(range(2))
    
    return action, model_prediction

def create_input_vector(current_state):
    
    input_vector = [0]*5
    input_vector[current_state] = 1
    return input_vector
    

#update the state of the environment, depending upon the action taken by 
#the agent
def perform_action(currentState, action):

    newState = currentState
    termination = False
    score = 0
    
    if action == 0 and currentState > 0: #left
        newState -= 1
    elif action == 1 and currentState < len(states)-1: #right
        newState += 1
    
    if action == 1 and currentState == len(states)-1: #termination condition
        termination = True
        score = 1
        
    return newState, termination, score

"""    
Remember the state for the current step in the game, the action that was taken
from here, and the prediction that the model made for the q_values (action choices)
(if the agent moved at random)    
"""
def store_action_in_game_memory(state, action, predicted_q_values, game_memory):
    state_result = {}
    state_result['state'] = create_input_vector(state)
    state_result['action'] = action
    state_result['Q_values'] = list(predicted_q_values[:])
    game_memory.append(state_result)
 
    
"""Go backwards through the game history, updating the q_value for the choice
that was made according to the utility that was gained by this move
i.e. For the action that scored a point, the q_value is the value of
that point. For the step just before this, the q_value is the value of 
the point multiplied by the discount_rate, and so on back to the first move
of the game
"""
def update_q_values_for_this_game(score, game_memory):

    discount_rate = 0.97
    
    for x in range(len(game_memory)-1, -1, -1):
        action = game_memory[x]['action']
        game_memory[x]['Q_values'][action] = score
        score *= discount_rate


def train_network(model, replay_memory):
    x_train, y_train, x_test, y_test = create_training_data(replay_memory)

    model.fit(x_train, y_train,
              batch_size=16,
              epochs=50,
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
    
    x_train = x_data[:split_position]
    x_test = x_data[split_position:]
    y_train = y_data[:split_position]
    y_test = y_data[split_position:]
    
    return x_train, y_train, x_test, y_test


def play_game_using_model(model, initial_state=0):
    
    terminate = False
    state = initial_state
    
    while not terminate:
        draw_environment(state)
        action, _ = choose_action(state, model)
        state, terminate, _ = perform_action(state, action)        
        time.sleep(1)
        
    
    
#main()