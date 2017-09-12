# -*- coding: utf-8 -*-
"""

Q_Learning example
Created on Tue Sep  5 11:39:54 2017

@author: brucew
"""

import numpy as np
import time


actions = ['left', 'right']
states = list(range(5))
    
def main():

    current_state = 0 
    learning_rate = 0.1
    
    action_table, visits_table = build_action_table(states, actions)
    n_loops = 0
    
    while True:
        action = choose_action(current_state, action_table)
        previous_state = current_state
        current_state, terminate = perform_action(current_state, action)
        reward = calculate_reward_for_state(action_table, current_state)
        
        update_visits_table(visits_table, previous_state, action)       
        update_action_table(action_table, visits_table, previous_state, action, reward, learning_rate)
        
        if terminate:
            current_state = 0
        
        draw_environment(current_state)
#        print(action_table)
#        print(visits_table)
        
        
        n_loops += 1
        
        if n_loops % 5 == 0:
            learning_rate = learning_rate * 0.95
                    
        time.sleep(.1)

def draw_environment(currentState):
    
    world = ''    
    for _ in range(0, currentState):
        world += ' _'    
    world = world + ' X'    
    for _ in range(currentState+1, len(states)):
        world += ' _'
    
    print(world)#, end='\r', flush=True)

def build_action_table(states, actions):

    action_table = np.zeros((len(states), len(actions)))    
    visits_table = np.ones((len(states), len(actions))) * 0.1

    # leaving the end position is the goal
    action_table[len(states)-1][len(actions)-1] = 5
    return action_table, visits_table


def choose_action(currentState, action_table):
    
    action = np.argmax(action_table[currentState])
    
    return action


def perform_action(currentState, action):

    newState = currentState
    termination = False
    
    if action == 0 and currentState > 0: #left
        newState -= 1
    elif action == 1 and currentState < len(states)-1: #right
        newState += 1
    
    if action == 1 and currentState == len(states)-1: #termination condition
        termination = True
        
    return newState, termination

    
def calculate_reward_for_state(action_table, state):

    #the expected reward for a state is the maximum of all its action rewards    
    reward = np.max(action_table[state])
    return reward        

    
def update_action_table(action_table, visits_table, previous_state, action, reward, learning_rate):
    old_action_value = action_table[previous_state][action]
    
    updated_action_value = ((1.0-learning_rate) * old_action_value) + (learning_rate * reward)
    exploration_bonus = -.1 / visits_table[previous_state][action]
    
    action_table[previous_state][action] = updated_action_value + exploration_bonus


def update_visits_table(visits_table, previous_state, action):
    visits_table[previous_state][action] += 1
    
    
main()