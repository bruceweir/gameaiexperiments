import keyboard
import time
import json
import cv2
import mss
import numpy as np
import pickle

key_states = [0, 0, 0, 0]
keep_capturing = True


def handle_keypress(keyevent):

    global keep_capturing
    global key_states

    key_value = 0

    if keyevent.event_type == 'down':
        key_value = 1

    if keyevent.name == 'a':
        key_states[0] = key_value
    elif keyevent.name == 'd':
        key_states[1] = key_value
    elif keyevent.name == 'w':
        key_states[2] = key_value
    elif keyevent.name == 's':
        key_states[3] = key_value
    elif keyevent.name == 'q':
        keep_capturing = False


keyboard.on_press_key('a', handle_keypress)
keyboard.on_release_key('a', handle_keypress)
keyboard.on_press_key('d', handle_keypress)
keyboard.on_release_key('d', handle_keypress)
keyboard.on_press_key('w', handle_keypress)
keyboard.on_release_key('w', handle_keypress)
keyboard.on_press_key('s', handle_keypress)
keyboard.on_release_key('s', handle_keypress)

keyboard.on_press_key('q', handle_keypress)

training_data = []

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 40, "left": 0, "width": 300, "height": 200}

    while keep_capturing:
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        
        img = np.array(sct.grab(monitor))
        # cv2.imshow("OpenCV/Numpy normal", img)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        print(key_states)
        training_data.append((img, np.array(key_states)))
        # Display the picture
        
        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

        #print("fps: {}".format(1 / (time.time() - last_time)))

    print(len(training_data))

np.save('training_data', training_data)

print("finished")
    