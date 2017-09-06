# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 21:49:18 2017

@author: brucew
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
import pyautogui
import math
from getkeys import key_check
import os


def keys_to_output(keys):
    #[A, W, D]
    output = [0, 0, 0]
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    
    return output

file_name = 'training_data.npy'    

if os.path.isfile(file_name):
    print('File exists, loading previous data')
    training_data = list(np.load(file_name))
else:
    print('Starting new trainind data file')    
    training_data = []
    
    
def main():
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
        
    last_time = time.time() 
    
    
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 600)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (80, 60))
        
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen, output])
                
        print(time.time() - last_time)        
        last_time = time.time()
        
        if(len(training_data) % 500 == 0):
            print(len(training_data))
            np.save(file_name, training_data)
            
        #cv2.namedWindow('window', cv2.WINDOW_NORMAL)
        #cv2.imshow('window', screen )
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        #    break
        
main()
