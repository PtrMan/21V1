# supposed to simulate primitive robot in a virtual 2.5 environment


class Scene(object):
    def __init__(self):
        self.cameraPos = [0.0, 0.0, 0.0]

# generate povray scene, write it to file, render file
# write png to local file
def renderScene(scene, displaySize):
    sceneTemplate = """
    #include "colors.inc"    

background { color Black }

camera {
  location <CAMERAPOS>
  look_at <0, 1, 2>
}

sphere {
  <0, 1, 40>, 2
  texture {
    pigment { color Yellow }
  }
}

light_source { <2, 4, -3> color White}
"""
    sceneContent = sceneTemplate[:]
    sceneContent = sceneContent.replace("CAMERAPOS", str(scene.cameraPos[0])+","+str(scene.cameraPos[1])+","+str(scene.cameraPos[2]))

    f = open("TEMPScene.pov", 'w')
    f.write(sceneContent)
    f.close()

    import os
    os.system("povray TEMPScene.pov"+" +W"+str(displaySize[0])+" +H"+str(displaySize[1]))


import cv2

def main():
    import pygame

    t = 0.0 # time

    scene = Scene() # create scene


    pygame.init()


    displaySize = (400, 300)


    gameDisplay = pygame.display.set_mode(displaySize)
    pygame.display.set_caption('ENV')

    while True: # "game"loop
        t+=0.3 # advance time

        scene.cameraPos = [0.0+(t % 2.0), 2.0, -3.0]
        
        # iterate over the list of Event objects 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # deactivates the pygame library 
                pygame.quit() 
  
                # quit the program. 
                quit() 

        black = (0, 0, 0)
        gameDisplay.fill(black) # fill background


        renderScene(scene, displaySize) # render scene with renderer

        image = pygame.image.load("TEMPScene.png")     
        gameDisplay.blit(image, (0, 0))


        # compute bounding box of object by synthetic mask
        #
        # we use a synthetic mask because we want to focus effort on UL with prototypes,
        # not generation of proposal BB from natural images
        #
        #
        # see https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
        img = cv2.imread("TEMPScene.png")
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        del img
        ret, imgBinary = cv2.threshold(imgGray,127,255,cv2.THRESH_BINARY)
        del imgGray
        synMaskRect = cv2.boundingRect(imgBinary)

        # draw bounding lines
        pygame.draw.rect(gameDisplay, (255,0,0,127), (synMaskRect[0], synMaskRect[1], synMaskRect[2], 3), width=0, border_radius=0)
        pygame.draw.rect(gameDisplay, (255,0,0,127), (synMaskRect[0], synMaskRect[1]+synMaskRect[3], synMaskRect[2], 3), width=0, border_radius=0)

        # crop by BB
        print("TODO - crop by BB")


        # Draws the surface object to the screen
        pygame.display.update()

main()