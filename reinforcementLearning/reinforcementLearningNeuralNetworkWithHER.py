# -*- coding: utf-8 -*-
import termplot
import time
import random
import os.path
import matplotlib.pyplot as plt
import numpy as np
import csv
import copy
import argparse
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Reshape, Dropout, Flatten, Activation, Input, concatenate
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import activations

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

ipython reinforcementLearningNeuralNetworkWithHER.py

This will train a neural network to search for a square in the play area
"""
"""
WORK IN PROGRESS - NOT YET FULLY FUNCTIONAL
https://youtu.be/ggqnxyjaKe4?t=2252
"""

parser = argparse.ArgumentParser(description='Grid-based reinforcement learning experiment. Can the agent find the target given a play area of a particular size?',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-x', '--width', dest='width', help='Play area width', type=int, default=5)
parser.add_argument('-y', '--height', dest='height', help='Play area height', type=int, default=5)
parser.add_argument('-t', '--training_runs', dest="max_training_runs", help='Number of training runs', type=int, default=100)
parser.add_argument('-g', '--games_per_training_run', dest="games_per_training_run", help='Number of games to play with an agent before training the network on his results', type=int, default=75)
parser.add_argument('-gl', '--game_step_limit', dest="game_step_limit", help='Maximum number of steps before a game is terminated', type=int, default=25)
parser.add_argument('-eps', '--epsilon_start', dest="epsilon", help='Initial value of epsilon (chance of taking greedy action)', type=float, default=1.0)
parser.add_argument('-epd', '--epsilon_decay', dest="epsilon_decay", help='Reduce epsilon by this much after each training run', type=float, default=0.02)
parser.add_argument('-epm', '--epsilon_minimum', dest="epsilon_minimum", help='Minimum limit of epsilon during training process', type=float, default=0.1)
parser.add_argument('-e', '--experiment_name', dest="experiment_name", help='Name of experiment. Results file will be saved as [experiment_name].csv. Best model file will be saved as [experiment_name].h5', default="experiment")
parser.add_argument('-her', '--hindsight_experience_replay', dest="hindsight_experience_replay", help='Use Hindsight Experience Replay to generate artificial games when they fail to reach a termination goal', default=False)
parser.add_argument('-m', '--model_file', dest="model_file", help='Path to a model file. If used then the game is played using the agent contained in the model file rather than training a new agent', default=None)

args = parser.parse_args()

print(vars(args))
# set this to False if you are running on a terminal with no graphic support
draw_graphics_with_matplot = False

if draw_graphics_with_matplot:
        fig = plt.figure()
        axes = fig.gca()
        fig.show()
        plt.ion()

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

random.seed(time.time())

actions = ['left', 'right', 'up', 'down']

width = args.width
height = args.height

path = './log'
num_files = len(os.listdir(path))

tensorBoard = TensorBoard(log_dir='./log/%d' % num_files,
                          histogram_freq=1, write_graph=True, write_images=True)
earlyStopping = EarlyStopping(patience=3)

success_score = 1.0
collision_score = -1.0
failure_to_finish_score = 0.0

trial_results = [['Collisions', 'Failures', 'Successes']]

if os.path.exists('gameslog.txt'):
    os.remove('gameslog.txt')


def learn_to_play(max_training_runs=100, use_Hindsight_Experience_Replay=True):

    (player_position, target_position) = generate_start_positions()

    initial_target_position = target_position

    environment = make_environment(player_position, target_position)
    # He will move entirely randomly at first
    epsilon_greedy = args.epsilon

    epsilon_decay = args.epsilon_decay

    epsilon_minimum = args.epsilon_minimum
    # Each step in a single game is recorded in the game_memory
    game_memory = []
    # The replay memory holds the steps of every game until a training run is needed
    replay_memory = []

    model = create_neural_network()
    
    best_test_score = 0.0

    n_games = 0

    n_training_runs = 0

    n_turns_in_this_game = 0

    game_turn_limit = args.game_step_limit

    number_of_games_to_play_before_training_run = args.games_per_training_run

    run = True

    while run:

        terminate_game = False
        record_game = True
        generate_Hindsight_Experience_Replay_episode = False

        # pick an action to perform

        action, model_q_predictions = choose_action(environment, model, epsilon_greedy)
        previous_environment = np.copy(environment)

        #  update the state of the environment, and return a score and (if the agent reaches
        #  the target, or has completed its max number of steps) a termination trigger
        player_position, terminate_game, score = perform_action(player_position, target_position, action)

        environment = make_environment(player_position, target_position)

        # draw_array(environment)

        # record the move that was taken
        loop_detected = store_action_in_game_memory(previous_environment, action, model_q_predictions, game_memory)

        n_turns_in_this_game += 1

        if n_turns_in_this_game == game_turn_limit:

            terminate_game = True

            if use_Hindsight_Experience_Replay:
                generate_Hindsight_Experience_Replay_episode = True
            else:
                #  print("***********Failed************")
                record_game = False

        if terminate_game:

            # print('Completed game ', n_games, ' in ', n_turns_in_this_game, ' steps.')

            if generate_Hindsight_Experience_Replay_episode:

                # print("Generating Hindsight Experience Replay episode")
                copy_of_game_memory = copy.deepcopy(game_memory)

                hindsight_memory = generate_Hindsight_memory(copy_of_game_memory, player_position)

                update_q_values_for_this_game(success_score, hindsight_memory)

                replay_memory.append(hindsight_memory[:])

            # go backwards through the game history and update the q_values to
            # reflect the score received

            if record_game:
                update_q_values_for_this_game(score, game_memory)
                replay_memory.append(game_memory[:])

            game_memory = []

            (player_position, target_position) = generate_start_positions()
            environment = make_environment(player_position, target_position)

            n_games += 1
            n_turns_in_this_game = 0

            if n_games == number_of_games_to_play_before_training_run:
                # once enough games have been played, train the network with the
                # game states as input data, and the measured q_values as the target
                print('Training run: %d, epsilon_greedy: %f' %
                      (n_training_runs, epsilon_greedy))
                train_network(model, replay_memory)

                # plot_training_log()

                replay_memory = []
                n_games = 0
                n_training_runs += 1

                epsilon_greedy = max([epsilon_greedy - epsilon_decay, epsilon_minimum])

                trial_result = test_model_performance(model, 100, 20)
                trial_results.extend([[trial_result['Collisions'], trial_result['Failures'], trial_result['Successes']]])
                write_as_csv(args.experiment_name+".csv", trial_results)

                test_score = trial_result['Successes'] / 100.0

                if test_score > best_test_score:
                    print("New best score: ", test_score)
                    best_test_score = test_score
                    model.save(args.experiment_name+".h5")

                print("Current score: {0}, best score: {1}".format(test_score, best_test_score))

        if n_training_runs == max_training_runs:
            run = False

    model.save('last_model.h5')
    return model


def generate_start_positions():

    player_position = [random.choice(range(width)), random.choice(range(height))]
    target_position = [random.choice(range(width)), random.choice(range(height))]

    while target_position == player_position:
        target_position = [random.choice(range(width)), random.choice(range(height))]

    return (player_position, target_position)


def make_environment(player_position, target_position):

    environment = np.zeros((width, height))

    environment[target_position[0]][target_position[1]] = -1

    environment[player_position[0]][player_position[1]] = 1

    return environment


def draw_array(environment):

    string_array = np.copy(environment)
    string_array = np.where(string_array == 1, 'p', string_array)
    string_array = np.where(string_array == '-1.0', 'X', string_array)
    string_array = np.where(string_array == '0.0', '', string_array)
    print(string_array)
    print()


def create_neural_network():

    mid_layer_activation = "elu"

    input_layer = Input(shape=(width*height,))

    conv_side = Reshape((width, height, 1))(input_layer)
    conv_side = Conv2D(filters=16, kernel_size=(2, 2), padding="same", activation=mid_layer_activation)(conv_side)
    conv_side = Conv2D(filters=4, kernel_size=(2, 2), padding="same", activation=mid_layer_activation)(conv_side)
    conv_side = Flatten()(conv_side)

    linear_side = Dense(height*width*2, activation=mid_layer_activation)(input_layer)
    linear_side = Dense(height*width*2, activation=mid_layer_activation)(linear_side)

    concatenated = concatenate([conv_side, linear_side])

    final = Dense(len(actions), activation='linear')(concatenated)    

    model = Model(inputs=input_layer, outputs=final)

    model.compile(optimizer='adam', loss='logcosh', metrics=[])

    # model = Sequential()

    # model.add(Dense((height)*(width), activation='linear', input_shape=(width*height,)))
    # model.add(Dense((height)*(width), activation='linear'))
    # model.add(Dense((height)*(width), activation='linear'))
    # model.add(Dropout(rate=0.2))
    # model.add(Dense(len(actions), activation='linear'))

    # model.compile(optimizer='adam',
    #              loss='mean_squared_error', metrics=[])

    return model


# Either choose the action that the model predicts is best, or a
# random action, depending upon the value of epsilon_greedy
def choose_action(environment, model, epsilon_greedy=0.0):

    action = 0
    # model_prediction = list(model.predict(np.array([environment.reshape(width, height, 1)]))[0][0][0])

    model_prediction = model.predict(np.array([np.reshape(environment, (width*height))]))[0]  # np.array([environment.reshape(width, height, 1)]))[0])

    if random.random() >= epsilon_greedy:
        action = np.argmax(model_prediction)
    else:
        action = random.choice(range(len(actions)))

    return action, model_prediction

# update the state of the environment, depending upon the action taken by
# the agent


def perform_action(player_position, target_position, action):

    termination = False
    score = failure_to_finish_score

    if action == 0:
        if player_position[1] > 0:  # left
            player_position[1] -= 1
        else:
            score = collision_score
            # print('************COLLISION*****************')
            termination = True

    elif action == 1:
        if player_position[1] < width-1:  # right
            player_position[1] += 1
        else:
            score = collision_score
            # print('************COLLISION*****************')      
            termination = True

    elif action == 2:
        if player_position[0] > 0:  # up
            player_position[0] -= 1
        else:
            score = collision_score
            # print('************COLLISION*****************')
            termination = True

    elif action == 3:
        if player_position[0] < height-1:  # down
            player_position[0] += 1
        else:
            score = collision_score
            # print('************COLLISION*****************')
            termination = True

    if player_position == target_position:
        termination = True
        # print('!!!!!!!!!SUCCESS!!!!!!!!!')
        score = success_score

    return player_position, termination, score


"""
Remember the state for the current step in the game, the action that was taken
from here, and the prediction that the model made for the q_values (action choices)
(if the agent moved at random)

Check that the agent is not in a repeating loop
"""


def store_action_in_game_memory(environment, action, predicted_q_values, game_memory):
    state_result = {}
    state_result['state'] = np.copy(environment.reshape(height*width))
    state_result['action'] = action
    state_result['Q_values'] = list(predicted_q_values[:])
    game_memory.append(state_result)

    loop_detected = False

    if len(game_memory) < 3:
        loop_detected = False
    elif np.array_equal(game_memory[-1]['state'], game_memory[-3]['state']):
            loop_detected = True

    return loop_detected


"""Go backwards through the game history, updating the q_value for the action
that was made according to the utility that was gained by this action
i.e. For the action that scored a point, the q_value is the value of
that point. For the step just before this, the q_value is the value of
the point multiplied by the discount_rate, and so on back to the first move
of the game
"""


def update_q_values_for_this_game(score, game_steps):

    discount_rate = 0.9

    gamma = .33

    for x in reversed(range(len(game_steps))):
        action = game_steps[x]['action']

        # print(game_memory[x])

        game_steps[x]['Q_values'][action] = (gamma * score) + ((1.0-gamma) * game_steps[x]['Q_values'][action])
        score *= discount_rate


def generate_Hindsight_memory(game_memory_copy, final_player_position):

    # assume that the final_player_position was actually where we
    # wanted him to be all along

    fake_target_position = final_player_position

    hindsight_memory = []

    for step in game_memory_copy:
        try:
            player_position = get_player_position_from_environment(step['state'])
            invented_state = make_environment(player_position, fake_target_position)

            hindsight_step = {'state': np.copy(invented_state.reshape(height*width)), 'action': step['action'], 'Q_values': step['Q_values']}
            hindsight_memory.append(hindsight_step)
        except ValueError:
            print("Player not found in environment. This is probably an error")
            continue

    return hindsight_memory


def get_player_position_from_environment(environment):

    if environment.shape != (height, width):
        environment = environment.reshape((height, width))

    position = np.where(environment == 1)
    
    if len(position[0]) == 0:
        raise ValueError("Player not found")

    return (position[0][0], position[1][0]) 


def train_network(model, replay_memory):
    x_train, y_train, x_test, y_test = create_training_data(replay_memory)

    model.fit(x_train, y_train,
              batch_size=50,
              epochs=40,
              verbose=False,
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

#    write_to_training_log(str(len(x_train) + len(x_test)))

    return x_train, y_train, x_test, y_test


def write_to_training_log(line):

    f = open('gameslog.txt', 'a', newline='\r\n')
    f.write(line + '\r\n')
    f.close()


def write_as_csv(logfilename, results):

    with open(logfilename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)


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


def test_model_performance(model, number_of_trials, turn_limit=20):

    collisions = 0.0
    successes = 0.0
    failures = 0.0

    for _ in range(number_of_trials):

        (player_position, target_position) = generate_start_positions()
        environment = make_environment(player_position, target_position)

        terminate_game = False
        turns_played = 0
        score = 0

        while not terminate_game and turns_played < turn_limit:

            action, model_q_predictions = choose_action(environment, model)
            previous_environment = np.copy(environment)

            #  update the state of the environment, and return a score and (if the agent reaches
            #  the target, or has completed its max number of steps) a termination trigger
            player_position, terminate_game, score = perform_action(player_position, target_position, action)

            environment = make_environment(player_position, target_position)

            turns_played += 1

        if score == collision_score:
            # print("Game ended by collision")
            collisions += 1
        elif score == failure_to_finish_score:
            # print("Game ended by failure")
            failures += 1
        elif score == success_score:
            # print("Game ended by success")
            successes += 1

    return {"Collisions": collisions,
            "Failures": failures,
            "Successes": successes}


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
    (player_position, target_position) = generate_start_positions()
    environment = make_environment(player_position, target_position)

    if draw_graphics_with_matplot:
        draw_environment(environment, fig, axes)
    else:
        print(environment)

    while not terminate_game:
        action, _ = choose_action(environment, model)
        player_position, terminate_game, score = perform_action(player_position, target_position, action)
        environment = make_environment(player_position, target_position)

        if draw_graphics_with_matplot:
            if user_has_closed_figure():
                print('Figure closed by user')
                terminate_game = True
            else:
                draw_environment(environment, fig, axes)

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
    plt.show()


draw_environment.initialized = False


def main():

    if args.model_file is None:
        learn_to_play(args.max_training_runs, args.hindsight_experience_replay)
    else:
        model = load_model(args.model_file)
        play_game_using_model(model)


if __name__ == "__main__":
    main()
