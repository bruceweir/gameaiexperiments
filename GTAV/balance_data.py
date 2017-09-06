# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 21:44:59 2017

@author: brucew
"""

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('training_data.npy')

df = pd.DataFrame(train_data)
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []

shuffle(train_data)

for data in train_data:
    image = data[0]
    choice = data[1]
    if choice == [1, 0, 0]:
        lefts.append([image, choice])
    elif choice == [0, 1, 0]:
        forwards.append([image, choice])
    elif choice == [0, 0, 1]:
        rights.append([image, choice])
#only keep as many samples as required to keep data balanced
#(this probably isn't a very good idea - better to replicate lesser samples)

forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(rights)]

final_data = forwards + lefts + rights

shuffle(final_data)
print(len(final_data))
np.save('training_data_balanced.npy', final_data)
    
    
#for data in train_data:
#    image = data[0]
#    choice = data[1]
#    cv2.imshow('test', image)
#    print(choice)
#    if cv2.waitKey(0) & 0xFF == ord('q'):
#        cv2.destroyAllWindows()
#        break
#    
#    
