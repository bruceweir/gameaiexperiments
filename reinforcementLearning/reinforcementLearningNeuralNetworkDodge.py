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
from keras.layers import Dense, Conv2D, MaxPooling2D, Reshape, Dropout, Flatten
from keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os.path
import random
import time

actions = ['left', 'right', 'wait']
width = 5
height = 10

path = './log'
num_files = len(os.listdir(path))

tensorBoard = TensorBoard(log_dir='./log/%d'%num_files, histogram_freq=1, write_graph=True, write_images=True)
earlyStopping = EarlyStopping(patience=3)


def learn_to_play(number_of_rocks=1):

    
    characters = create_game_characters(number_of_rocks)

    environment = make_environment(characters)
    #He will move entirely randomly at first
    epsilon_greedy = 1.0
    #Each step in a single game is recorded in the game_memory
    game_memory = []
    #The replay memory holds the steps of every game until a training run is needed
    replay_memory = []
    model = create_neural_network()
    n_games = 0
    
    max_training_runs = 10
    n_training_runs = 0

    n_turns_in_this_game=0
    
    run = True

    while run:
        #pick an action to perform

        action, model_q_predictions = choose_action(environment, model, epsilon_greedy)
        previous_environment = np.copy(environment)
        #update the state of the environment, and return a score and (if the agent has left the right hand edge of
        #the environment) a termination trigger
        environment, characters, terminate_game, score = perform_action(environment, characters, action)
        #record the move that was taken
        store_action_in_game_memory(previous_environment, action, model_q_predictions, game_memory)
        #print(environment)
        #print(terminate_game)
        n_turns_in_this_game += 1
        
        if terminate_game or n_turns_in_this_game == 1000:

            print('Completed game ', n_games)
            #go backwards through the game history and update the q_values to
            #reflect the score received
            update_q_values_for_this_game(score, game_memory)
            replay_memory.append(game_memory[:])
            game_memory = []
            characters = create_game_characters(number_of_rocks)
            environment = make_environment(characters)
            n_games += 1
            n_turns_in_this_game=0

            if n_games == 500:
                #once enough games have been played, train the network with the
                #game states as input data, and the measured q_values as the target
                print('Training run: %d' % n_training_runs)
                train_network(model, replay_memory)
                replay_memory = []
                n_games = 0
                n_training_runs += 1
                epsilon_greedy *= 0.05 ** (1/max_training_runs) # seems to help if this is down to < 0.05 by the final training run

        if n_training_runs == max_training_runs:
            run = False

    model.save('last_model.h5')
    return model
            
 #       time.sleep(.1)

def create_game_characters(n_rocks=1):
    
    player_start = random.choice(range(width))
    
    character_status = {'player': player_start}
    character_status['number_of_rocks'] = n_rocks
    character_status['rocks'] = []
    
    for _ in range(n_rocks):
        rock_start = random.choice(range(width))
        character_status['rocks'].append([rock_start, random.choice(range(4))])    
    
    return character_status
    
def make_environment(characters):
    
    environment = np.zeros((height, width))
    environment[height-1][characters['player']] = 1
    
    rocks = characters['rocks']
    
    for r in rocks:
        environment[r[1]][r[0]] = 1
        
    return environment


def create_neural_network():
    
    model = Sequential()
    model.add(Dense(height*width, input_shape=(height*width,)))
    model.add(Dense(int(height*width/3), activation='elu'))
    model.add(Dense(len(actions), activation='tanh')) 
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
def perform_action(environment, characters, action):
    
    termination = False
    score = .1
    
    characters = move_rocks(characters)
    
    if action == 0 and characters['player'] > 0: #left
        characters['player'] -=1
    elif action == 0 and characters['player'] == 0: #left
        characters['player'] = width-1 #loop round
    elif action == 1 and characters['player'] < width-1: #right
        characters['player'] += 1
    elif action == 1 and characters['player'] == width-1: #right
        characters['player'] = 0 #loop around
        
        
    
     
    for r in characters['rocks']:
        if r[0] == characters['player'] and r[1] == height-1:
            termination = True
            score = -1
            break
    
    environment = make_environment(characters)
    return environment, characters, termination, score

def move_rocks(characters):
    
    rocks_to_remove = []
    for r in reversed(range(len(characters['rocks']))):
        characters['rocks'][r][1] += 1
        if characters['rocks'][r][1] >= height:
            rocks_to_remove.append(r)
    
    for r in rocks_to_remove:
        characters['rocks'].pop(r)
    
    
    while len(characters['rocks']) < characters['number_of_rocks']:
        characters['rocks'].append([random.choice(range(width)), random.choice(range(3))])
 
    return characters
            
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
              epochs=20,
              verbose=True,
              callbacks=[tensorBoard, earlyStopping],
              validation_data=(x_test, y_test),
              shuffle=True)    
    

"""
The training inputs are the game state and target outputs are the
q_values which we have been measuring as we have been playing the game
"""
def create_training_data(replay_memory):
    
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


def play_game_using_model(model, number_of_rocks=1):
    
    f = plt.figure()
    ax = f.gca()
    f.show()
    plt.ion()
    
    terminate_game = False
    characters = create_game_characters(number_of_rocks)    
    environment = make_environment(characters)
    draw_environment(environment, f, ax)
    
    while not terminate_game:
        action, _ = choose_action(environment, model)
        environment, characters, terminate_game, _ = perform_action(environment, characters, action)        
#        draw_environment(environment, f, ax)
        redraw_fn(environment, f, ax)
        time.sleep(.01)

    redraw_fn.initialized = False


def draw_environment(environment, figure, axes):
    
    axes.imshow(environment)
    figure.canvas.draw()
#    plt.pause(0.01)
#    plt.show()
    
    print(environment, end='\r\r')
    
        
    #draw_environment(environment)
def redraw_fn(environment, figure, axes):
    if not redraw_fn.initialized:
        redraw_fn.im = axes.imshow(environment, animated=True)
        redraw_fn.initialized = True
    else:
        redraw_fn.im.set_array(environment)
    plt.pause(0.01)
redraw_fn.initialized = False
    
#main()
