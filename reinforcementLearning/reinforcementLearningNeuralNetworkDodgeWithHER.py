# -*- coding: utf-8 -*-
import termplot
import time
import random
import os.path
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Reshape, Dropout, Flatten
from tensorflow.python.keras.models import Sequential, load_model
"""
Created on Fri Sep  8 10:01:42 2017

@author: bruceweir1.bw@googlemail.com

Released without restriction. Use at own risk.
"""

# -*- coding: utf-8 -*-
"""
A simple game showing how a neural network can learn q_values (state/action choices) with additional
Hindsight Experience Replay - https://arxiv.org/pdf/1707.01495.pdf

Assuming that you have all the dependencies installed (Tensorflow, keras etc)
then run the application using:

ipython reinforcementLearningNeuralNetworkDodge.py

This will train a neural network to search for a square in the play area
"""


# set this to False if you are running on a terminal with no graphic support
draw_graphics_with_matplot = True
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

actions = ['left', 'right', 'up', 'down']
player_position = [0, 0]
target_position = [0, 0]

width = 5
height = 5

path = './log'
num_files = len(os.listdir(path))

tensorBoard = TensorBoard(log_dir='./log/%d' % num_files,
                          histogram_freq=1, write_graph=True, write_images=True)
earlyStopping = EarlyStopping(patience=3)

if os.path.exists('gameslog.txt'):
    os.remove('gameslog.txt')


def learn_to_play(max_training_runs=100, model_file=None):

    create_game_characters()

    environment = make_environment()
    # He will move entirely randomly at first
    epsilon_greedy = 1
    # Each step in a single game is recorded in the game_memory
    game_memory = []
    # The replay memory holds the steps of every game until a training run is needed
    replay_memory = []

    if model_file is None:
        model = create_neural_network()
    else:
        model = load_model(model_file)
        epsilon_greedy = 0.01

    n_games = 0

    #max_training_runs = max_training_runs
    n_training_runs = 0

    n_turns_in_this_game = 0

    number_of_games_to_play = 25
    run = True

    while run:
        # pick an action to perform

        action, model_q_predictions = choose_action(environment, model, epsilon_greedy)
        previous_environment = np.copy(environment)

        # update the state of the environment, and return a score and (if the agent reaches the target, or has completed its max number of steps) a termination trigger
        environment, terminate_game, score = perform_action(environment, action)

        draw_array_as_string(environment)
        # record the move that was taken
        store_action_in_game_memory(previous_environment, action, model_q_predictions, game_memory)

        n_turns_in_this_game += 1

        if terminate_game or n_turns_in_this_game == 20:

            print('Completed game ', n_games, ' in ',
                  n_turns_in_this_game, ' steps.')
            # go backwards through the game history and update the q_values to
            # reflect the score received
            update_q_values_for_this_game(score, game_memory)
            replay_memory.append(game_memory[:])
            game_memory = []
            create_game_characters()
            environment = make_environment()
            n_games += 1
            n_turns_in_this_game = 0

            if n_games == number_of_games_to_play:
                # once enough games have been played, train the network with the
                # game states as input data, and the measured q_values as the target
                print('Training run: %d, epsilon_greedy: %f' %
                      (n_training_runs, epsilon_greedy))
                train_network(model, replay_memory)

                plot_training_log()

                replay_memory = []
                n_games = 0
                n_training_runs += 1
                # seems to help if this is down to < 0.05 by the final training run
                epsilon_greedy = max([epsilon_greedy - 0.02, 0.0001])

        if n_training_runs == max_training_runs:
            run = False

    model.save('last_model.h5')
    return model


def create_game_characters():

    global player_position, target_position

    player_position = [random.choice(range(width)), random.choice(range(height))]

    while target_position == player_position:
        target_position = [random.choice(range(width)), random.choice(range(height))]


def make_environment():

    global player_position, target_position
 
    environment = np.zeros((height, width))

    environment[player_position[0]][player_position[1]] = 1

    environment[target_position[0]][target_position[1]] = .5
        
    return environment


def draw_array_as_string(environment):

    string_array = np.full((width, height), [' '], dtype=str)
    string_array[player_position[0]][player_position[1]] = 'p'
    string_array[target_position[0]][target_position[1]] = 'X'
    print(string_array)
    print()


def create_neural_network():

    model = Sequential()
    model.add(Dense(height*width*4, input_shape=(height*width,)))
    model.add(Dense(len(actions), activation='tanh'))
    model.compile(optimizer='adam',
                  loss='mean_squared_error', metrics=[])

    return model


# Either choose the action that the model predicts is best, or a
# random action, depending upon the value of epsilon_greedy
def choose_action(environment, model, epsilon_greedy=0.0):

    action = 0
    model_prediction = list(model.predict(np.array([environment.reshape(width*height)]))[0])

    if random.random() > epsilon_greedy:
        action = np.argmax(model_prediction)
    else:
        action = random.choice(range(len(actions)))

    return action, model_prediction

# update the state of the environment, depending upon the action taken by
# the agent


def perform_action(environment, action):

    global player_position, target_position

    termination = False
    score = -1

    if action == 0 and player_position[0] > 0:  # left
        player_position[0] -= 1
    elif action == 1 and player_position[0] < width-1:  # right
        player_position[0] += 1
    elif action == 2 and player_position[1] < height-1:  # up
        player_position[1] += 1
    elif action == 3 and player_position[1] > 0: #down
        player_position[1] -= 1

    if player_position == target_position:
        termination = True
        score = 1

    environment = make_environment()
    return environment, termination, score



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

    discount_rate = 0.9

    gamma = 0.9

    for x in reversed(range(len(game_memory))):
        action = game_memory[x]['action']
        game_memory[x]['Q_values'][action] = (
            gamma * score) + ((1.0-gamma) * game_memory[x]['Q_values'][action])
        score *= discount_rate


def train_network(model, replay_memory):
    x_train, y_train, x_test, y_test = create_training_data(replay_memory)

    model.fit(x_train, y_train,
              batch_size=50,
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

    x_data = []
    y_data = []
    for game in replay_memory:
        x_data.extend([gamestep['state'] for gamestep in game])
        y_data.extend([gamestep['Q_values'] for gamestep in game])

    split_position = int(.9 * len(y_data))

    x_train = np.array(x_data[:split_position])
    x_test = np.array(x_data[split_position:])
    y_train = np.array(y_data[:split_position])
    y_test = np.array(y_data[split_position:])

    write_to_training_log(str(len(x_train) + len(x_test)))

    return x_train, y_train, x_test, y_test


def write_to_training_log(line):

    f = open('gameslog.txt', 'a', newline='\r\n')
    f.write(line + '\r\n')
    f.close()


def plot_training_log():

    games_played = np.loadtxt('gameslog.txt')

    if draw_graphics_with_matplot:

        plt.ion()
        plt.plot(games_played)
        plt.show()
        plt.pause(0.01)
    else:
        if len(games_played.shape) > 0:
            termplot.plot(games_played)


def play_game_using_model(model):

    def user_has_closed_figure():
        if len(plt.get_fignums()) == 0:  # check for user closing the rendering window
            return True
        return False

    if draw_graphics_with_matplot:
        f = plt.figure()
        ax = f.gca()
        f.show()
        plt.ion()

    terminate_game = False
    create_game_characters()
    environment = make_environment()

    if draw_graphics_with_matplot:
        draw_environment(environment, f, ax)
    else:
        print(environment)

    while not terminate_game:
        action, _ = choose_action(environment, model)
        environment, terminate_game, _ = perform_action(environment, action)

        if draw_graphics_with_matplot:
            if user_has_closed_figure():
                print('Figure closed by user')
                terminate_game = True
            else:
                draw_environment(environment, f, ax)

        else:
            print(environment)
            print()

        time.sleep(.1)

    draw_environment.initialized = False


def draw_environment(environment, figure, axes):
    if not draw_environment.initialized:
        draw_environment.im = axes.imshow(environment, animated=True)
        draw_environment.initialized = True
    else:
        if len(plt.get_fignums()) > 0:
            draw_environment.im.set_array(environment)

    figure.canvas.draw()
    plt.pause(0.01)


draw_environment.initialized = False


def main():

    model = learn_to_play(50)
    play_game_using_model(model)


if __name__ == "__main__":
    main()
