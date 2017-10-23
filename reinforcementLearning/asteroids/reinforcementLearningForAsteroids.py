# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:01:42 2017

@author: bruce.weir@bbc.co.uk

Released without restriction. Use at own risk.
"""

# -*- coding: utf-8 -*-
"""
Class for handling reinforcement learning with a neural network

"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Reshape, Dropout, Flatten, BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os.path
import random
import time


path = './log'
num_files = len(os.listdir(path))

tensorBoard = TensorBoard(log_dir='./log/%d'%num_files, histogram_freq=1, write_graph=True, write_images=True)
earlyStopping = EarlyStopping(patience=3)

if os.path.exists('gameslog.txt'):
    os.remove('gameslog.txt')


epsilon_greedy = 1

game_memory = []
replay_memory = []

actions = ['wait', 'turnLeft', 'turnRight', 'fire', 'forward']

def get_action_index(action):
    return actions.index(action)

def get_action_string(index):
    return actions[index]

#Either choose the action that the model predicts is best, or a
#random action, depending upon the value of epsilon_greedy
def choose_action(game_image):


    action = 0
    model_prediction = list(model.predict(np.array([game_image]))[0])

    if random.random() > epsilon_greedy:
        action = np.argmax(model_prediction)
    else:
        action = random.choice(range(len(actions)))

    #print('choose action ', get_action_string(action))

    return action, model_prediction

"""
Remember the state for the current step in the game, the action that was taken
from here, and the prediction that the model made for the q_values (action choices)
(if the agent moved at random)
"""
def store_action_in_game_memory(game_image, action, predicted_q_values):

    #print('store_action_in_game_memory: ', action, ' ' , predicted_q_values)

    global game_memory
    state_result = {}
    state_result['state'] = np.copy(game_image)
    state_result['action'] = get_action_index(action)
    state_result['Q_values'] = list(predicted_q_values[:])
    game_memory.append(state_result)


"""Go backwards through the game history, updating the q_value for the action
that was made according to the utility that was gained by this action
i.e. For the action that scored a point, the q_value is the value of
that point. For the step just before this, the q_value is the value of
the point multiplied by the discount_rate, and so on back to the first move
of the game
"""
def update_q_values_for_this_game(score):

    print('update_q_values_for_this_game ', score)

    global game_memory
    discount_rate = 0.97

    for x in reversed(range(len(game_memory))):
        action = game_memory[x]['action']
        game_memory[x]['Q_values'][action] = score
        score *= discount_rate

    add_q_values_to_full_history()


def add_q_values_to_full_history():

    global game_memory
    global replay_memory

    replay_memory.append(game_memory[:])
    game_memory = []


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

    split_position = int(.9  * len(y_data))

    x_train = np.array(x_data[:split_position])
    x_test = np.array(x_data[split_position:])
    y_train = np.array(y_data[:split_position])
    y_test = np.array(y_data[split_position:])

    write_to_training_log(str(len(x_train) + len(x_test)))

    return x_train, y_train, x_test, y_test


def train_network():

    global model
    global replay_memory

    x_train, y_train, x_test, y_test = create_training_data(replay_memory)

    model.fit(x_train, y_train,
              batch_size=16,
              epochs=2,
              verbose=True,
              callbacks=[tensorBoard, earlyStopping],
              validation_data=(x_test, y_test),
              shuffle=True)

    clear_memory()
    
"""

        action, model_q_predictions = choose_action(environment, model, epsilon_greedy)
        previous_environment = np.copy(environment)
        #update the state of the environment, and return a score and (if the agent and a rock collide) a termination trigger
        environment, characters, terminate_game, score = perform_action(environment, characters, action)
        #record the move that was taken
        store_action_in_game_memory(previous_environment, action, model_q_predictions, game_memory)

        n_turns_in_this_game += 1

        if terminate_game or n_turns_in_this_game == 1000:

            print('Completed game ', n_games, ' in ', n_turns_in_this_game, ' steps.')
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

                plot_training_log()

                replay_memory = []
                n_games = 0
                n_training_runs += 1
                epsilon_greedy = max([epsilon_greedy - 0.02, 0.0001])# seems to help if this is down to < 0.05 by the final training run


        if n_training_runs == max_training_runs:
            run = False

    model.save('last_model.h5')
    return model
"""

def clear_memory():
    global replay_memory, game_memory
    
    replay_memory = []
    game_memory = []
    

def create_neural_network():

    model = Sequential()
    model.add(Conv2D(32, (28, 28), activation='tanh', input_shape=(128, 128, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (28, 28), activation='tanh'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(len(actions), activation='tanh'))
    model.compile(optimizer='adam',
          loss='mean_squared_error', metrics=[])

    return model

model = create_neural_network()


def write_to_training_log(line):

    f = open('gameslog.txt', 'a', newline='\r\n')
    f.write(line + '\r\n')
    f.close()


def plot_training_log():

    plt.ion()
    games_played = np.loadtxt('gameslog.txt')
    plt.plot(games_played)
    plt.show()
    plt.pause(0.01)
