#game logic copied from the asteroids code at http://knuth.luther.edu/~leekent/SamplingCS/lesson11.html


from turtle import *
import tkinter.messagebox
import tkinter
import random
import math
import datetime

import pyscreenshot as ImageGrab
import numpy as np
import scipy.misc
import PIL
from matplotlib import pyplot as plt

import platform

from reinforcementLearningForAsteroids import *

using_windows = False
if platform.system() == 'Windows':
    from grabscreen import grab_screen
    using_windows = True

screenMinX = -500
screenMinY = -500
screenMaxX = 500
screenMaxY = 500

mode = 'Train'
last_action = 'wait'
first_pass = True
previous_game_image = np.zeros((128, 128, 1))

model_q_predictions = [0]*len(actions)

class PhotonTorpedo(RawTurtle):
    def __init__(self,canvas,x,y,direction,dx,dy):
        super().__init__(canvas)
        self.penup()
        self.goto(x,y)
        self.setheading(direction)
        self.color("Green")
        self.lifespan = 200
        self.dx = math.cos(math.radians(direction)) * 2 + dx
        self.dy = math.sin(math.radians(direction)) * 2 + dy
        self.shape("bullet")

    def getLifeSpan(self):
        return self.lifespan

    def getDX(self):
        return self.dx

    def getDY(self):
        return self.dy

    def getRadius(self):
        return 4

    def move(self):
        self.lifespan = self.lifespan - 1
        screen = self.getscreen()
        x = self.xcor()
        y = self.ycor()
        x = (self.dx + x - screenMinX) %  \
            (screenMaxX - screenMinX) + screenMinX
        y = (self.dy + y - screenMinY) % \
            (screenMaxY - screenMinY) + screenMinY

        self.goto(x,y)

class Asteroid(RawTurtle):
    def __init__(self,canvas,dx,dy,x,y,size):
        RawTurtle.__init__(self,canvas)
        self.penup()
        self.goto(x,y)
        self.size = size
        self.dx = dx
        self.dy = dy
        self.shape("rock" + str(size))
        self.rotation = random.random() * 5

    def getSize(self):
        return self.size

    def getDX(self):
        return self.dx

    def getDY(self):
        return self.dy

    def setDX(self,dx):
        self.dx = dx

    def setDY(self,dy):
        self.dy = dy

    def move(self):
        screen = self.getscreen()
        x = self.xcor()
        y = self.ycor()

        x = (self.dx + x - screenMinX) % (screenMaxX - screenMinX) + screenMinX
        y = (self.dy + y - screenMinY) % (screenMaxY - screenMinY) + screenMinY

        self.goto(x,y)
        self.setheading(self.heading()+self.rotation)

    def getRadius(self):
        return self.size * 15

class SpaceShip(RawTurtle):
    def __init__(self,canvas,dx,dy,x,y):
        RawTurtle.__init__(self,canvas)
        self.penup()
        self.color("#000000")
        self.goto(x,y)
        self.dx = dx
        self.dy = dy
        self.shape("ship")

    def move(self):
        screen = self.getscreen()
        x = self.xcor()
        y = self.ycor()

        x = (self.dx + x - screenMinX) % (screenMaxX - screenMinX) + screenMinX
        y = (self.dy + y - screenMinY) % (screenMaxY - screenMinY) + screenMinY

        self.goto(x,y)

    def fireEngine(self):
        angle = self.heading()
        x = math.cos(math.radians(angle))
        y = math.sin(math.radians(angle))
        self.dx = self.dx + x
        self.dy = self.dy + y

    def getRadius(self):
        return 2

    def getDX(self):
        return self.dx

    def getDY(self):
        return self.dy

def intersect(object1,object2):
        dist = math.sqrt((object1.xcor() - object2.xcor())**2 + (object1.ycor() - object2.ycor())**2)

        radius1 = object1.getRadius()
        radius2 = object2.getRadius()

        # The following if statement could be written as
        # return dist <= radius1+radius2
        if dist <= radius1+radius2:
            return True
        else:
            return False

def main():

    # Start by creating a RawTurtle object for the window.
    root = tkinter.Tk()
    root.title("Asteroids!")
    cv = ScrolledCanvas(root,600,600,600,600)
    cv.pack(side = tkinter.LEFT)
    t = RawTurtle(cv)

    screen = t.getscreen()
    #screen.bgcolor('black')
    screen.setworldcoordinates(screenMinX,screenMinY,screenMaxX,screenMaxY)
    screen.register_shape("rock3",((-20, -16),(-21, 0), (-20,18),(0,27),(17,15),(25,0),(16,-15),(0,-21)))
    screen.register_shape("rock2",((-15, -10),(-16, 0), (-13,12),(0,19),(12,10),(20,0),(12,-10),(0,-13)))
    screen.register_shape("rock1",((-10,-5),(-12,0),(-8,8),(0,13),(8,6),(14,0),(12,0),(8,-6),(0,-7)))
    screen.register_shape("ship",((-10,-10),(0,-5),(10,-10),(0,10)))
    screen.register_shape("bullet",((-2,-4),(-2,4),(2,4),(2,-4)))
    frame = tkinter.Frame(root)
    frame.pack(side = tkinter.RIGHT,fill=tkinter.BOTH)

    scoreVal = tkinter.StringVar()
    scoreVal.set("0")
    scoreTitle = tkinter.Label(frame,text="Score")
    scoreTitle.pack()
    scoreFrame = tkinter.Frame(frame,height=2, bd=1, \
        relief=tkinter.SUNKEN)
    scoreFrame.pack()
    score = tkinter.Label(scoreFrame,height=2,width=20,\
        textvariable=scoreVal,fg="Yellow",bg="black")

    score.pack()

    livesTitle = tkinter.Label(frame, \
       text="Extra Lives Remaining")
    livesTitle.pack()

    livesFrame = tkinter.Frame(frame, \
        height=30,width=60,relief=tkinter.SUNKEN)
    livesFrame.pack()
    livesCanvas = ScrolledCanvas(livesFrame,150,40,150,40)
    livesCanvas.pack()
    livesTurtle = RawTurtle(livesCanvas)
    livesTurtle.ht()
    livesScreen = livesTurtle.getscreen()
    livesScreen.register_shape("ship", \
        ((-10,-10),(0,-5),(10,-10),(0,10)))
    life1 = SpaceShip(livesCanvas,0,0,-35,0)
    life2 = SpaceShip(livesCanvas,0,0,0,0)
    life3 = SpaceShip(livesCanvas,0,0,35,0)
    lives = [life1, life2, life3]

    t.ht()

    def quitHandler():
        root.destroy()
        root.quit()

    quitButton = tkinter.Button(frame, text = "Quit", command=quitHandler)
    quitButton.pack()

    screen.tracer(0)

    def reset_game():
        
        global first_pass
        first_pass = True
        
        screen.clear()
        screen.tracer(0)
        
        ship = SpaceShip(cv,0,0,(screenMaxX-screenMinX)/2+screenMinX,(screenMaxY-screenMinY)/2 + screenMinY)
    
        asteroids = []
        bullets = []
    
        for k in range(5):
            dx = random.random() * 6 - 3
            dy = random.random() * 6 - 3
            x = random.random() * (screenMaxX - screenMinX) + screenMinX
            y = random.random() * (screenMaxY - screenMinY) + screenMinY
    
            asteroid = Asteroid(cv,dx,dy,x,y,3)
    
            asteroids.append(asteroid)
        
        return ship, asteroids, bullets
    
    ship, asteroids, bullets = reset_game()
    

    def play():
        # Tell all the elements of the game to move

        global last_action
        global first_pass
        global model_q_predictions
        global previous_game_image
        result_of_the_last_action = 0
        
        
        start = datetime.datetime.now()
        screen.update()
        game_over = False

        if len(asteroids) == 0:
            game_over = True

        ship.move()

        deadbullets = []

        for bullet in bullets:
            bullet.move()
            if bullet.getLifeSpan() <= 0:
                deadbullets.append(bullet)



        for asteroid in asteroids:
            asteroid.move()

        hitasteroids = []
        for bullet in bullets:
            for asteroid in asteroids:
                if not asteroid in hitasteroids and \
                             intersect(bullet,asteroid):
                    deadbullets.append(bullet)
                    hitasteroids.append(asteroid)
                    asteroid.setDX(bullet.getDX() + \
                      asteroid.getDX())
                    asteroid.setDY(bullet.getDY() + \
                      asteroid.getDY())

        for bullet in deadbullets:
            try:
                bullets.remove(bullet)
            except:
                print("didn't find bullet")

            bullet.goto(-screenMinX*2, -screenMinY*2)
            bullet.ht()

        for asteroid in hitasteroids:
            try:
                asteroids.remove(asteroid)
            except:
                print("didn't find asteroid in list")

            asteroid.ht()
            size = asteroid.getSize()

            score = int(scoreVal.get())

            if size == 3:
                score += 20
            elif size == 2:
                score += 50
            elif size == 1:
                score += 100

            scoreVal.set(str(score))
            result_of_the_last_action += 1

            if asteroid.getSize() > 1:
                dx = asteroid.getDX()
                dy = asteroid.getDY()
                dist = math.sqrt(dx ** 2 + dy ** 2)
                normDx = asteroid.getDX() / dist
                normDy = asteroid.getDY() / dist

                asteroid1 = Asteroid(cv,-normDy,normDx, \
                  asteroid.xcor(),asteroid.ycor(), \
                  asteroid.getSize()-1)
                asteroid2 = Asteroid(cv,normDy,-normDx, \
                  asteroid.xcor(),asteroid.ycor(), \
                  asteroid.getSize()-1)
                asteroids.append(asteroid1)
                asteroids.append(asteroid2)

        shipHitAsteroids = []
        shipHit = False
        

        for asteroid in asteroids:
            if intersect(asteroid,ship):
                result_of_the_last_action = -1
                if len(lives) > 0:
                    if not shipHit:
                        #tkinter.messagebox.showwarning( \
                        #     "Uh-Oh","You Lost a Ship!")
                        deadship = lives.pop()
                        deadship.ht()
                        shipHit = True
                    shipHitAsteroids.append(asteroid)
                else:
                    if mode == 'Train':
                        game_over = True
                    else:
                        tkinter.messagebox.showwarning("Game Over", \
                        "Your game is finished!\nPlease try again.")
                        return

        for asteroid in shipHitAsteroids:
            try:
                asteroids.remove(asteroid)
            except:
                print("asteroid that hit ship not found")


            asteroid.ht()
            asteroid.goto(screenMaxX*2, screenMaxY*2)
        # Set the timer to go off again in 5 milliseconds

        end = datetime.datetime.now()
        duration = end - start

        millis = duration.microseconds / 1000.0

        if mode == 'Train':
            game_image = None
            if using_windows:
                game_image = windows_screen_grab()
            else:
                game_image = get_screen_array()

            game_image = game_image - previous_game_image
            previous_game_image = np.copy(game_image)
            
            if not first_pass:
                store_action_in_game_memory(game_image, last_action, model_q_predictions)

                if len(hitasteroids) > 0 or len(shipHitAsteroids) > 0:
                    update_q_values_for_this_game(result_of_the_last_action)

                if game_over:
                   # train_network()
                    reset_game()


        if mode == 'Train':
            last_action, model_q_predictions = choose_action(game_image)
            last_action = get_action_string(last_action)

            if last_action == 'turnLeft':
                turnLeft()
            if last_action == 'turnRight':
                turnRight()
            if last_action == 'forward':
                forward()
            if last_action == 'fire':
                fire()


        first_pass = False

        screen.ontimer(play,int(10-millis))

    # Set the timer to go off the first time in 5 milliseconds
    screen.ontimer(play, 5)

    def turnLeft():
        ship.setheading(ship.heading()+7)

    def turnRight():
        ship.setheading(ship.heading()-7)

    def forward():
        ship.fireEngine()

    def fire():
        if len(bullets) < 5:
            bullet = PhotonTorpedo(cv,ship.xcor(),ship.ycor(), \
               ship.heading(),ship.getDX(),ship.getDY())
            bullets.append(bullet)


    screen.onkeypress(turnLeft,"a")
    screen.onkeypress(turnRight,"d")
    screen.onkeypress(forward,"w")
    screen.onkeypress(fire," ")


    def get_screen_array():

        screenshot = take_screenshot()
        screenshot = screenshot.resize((128, 128), resample=PIL.Image.BILINEAR)
        screenshot = screenshot.convert('L')
        screen_array = np.array(list(screenshot.getdata()), dtype='uint8')
        screen_array = np.reshape(screen_array, (128, 128, 1))

        return screen_array

    def take_screenshot():

        x, y, width, height = get_asteroids_canvas_coords()
        return ImageGrab.grab(bbox=(x, y, x+width, y+height))

    def get_asteroids_canvas_coords():
        x = root.winfo_rootx() + cv.winfo_x()
        y = root.winfo_rooty() + cv.winfo_y()
        width = cv.winfo_width()
        height = cv.winfo_height()

        return x, y, width, height

    def windows_screen_grab():

        x, y, width, height = get_asteroids_canvas_coords()
        screenshot = grab_screen(region=[x, y, x+width, y+width])
        screenshot = scipy.misc.imresize(screenshot, (128, 128))
        screenshot = np.reshape(screenshot, (128, 128, 1))

        return screenshot
        #plt.imshow(screenshot[:,:,0], cmap='gray')
        #plt.show()



    if using_windows:
        screen.onkeypress(windows_screen_grab, 'p')
    else:
        screen.onkeypress(get_screen_array, "p")


    screen.listen()
    tkinter.mainloop()

if __name__ == "__main__":
    main()
