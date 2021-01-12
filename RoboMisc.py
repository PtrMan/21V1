# miscollenious functions for SpongeBot robot setting

import math

# compute camera direction by angle of orientation of spongebot
def spongebotCalcCameraDirByAngle(angle):
    #return [1.0, 0.0, 0.0] # forward
    #return [0.0, -1.0, 0.0] # down

    return [math.cos(angle), -1.0, math.sin(angle)]

