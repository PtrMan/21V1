# supposed to simulate primitive robot in a virtual 2.5 environment


class Scene(object):
    def __init__(self):
        self.cameraPos = [0.0, 0.0, 0.0]
        self.lookAt = [0.0, 0.0, 1.0]

        self.lightPos = [0.0, 0.0, 0.0]

        self.enBox = True
        self.enSphere = True

# generate povray scene, write it to file, render file
# write png to local file
def renderScene(scene, displaySize):
    sceneTemplate = """
    #include "colors.inc"    

background { color Black }

camera {
  location <CAMERAPOS>
  look_at <LOOKAT>
}

OBJS

light_source { <LIGHTPOS> color White}
"""
    sceneContent = sceneTemplate[:]
    sceneContent = sceneContent.replace("CAMERAPOS", str(scene.cameraPos[0])+","+str(scene.cameraPos[1])+","+str(scene.cameraPos[2]))

    sceneContent = sceneContent.replace("LIGHTPOS",  str(scene.lightPos[0])+","+str(scene.lightPos[1])+","+str(scene.lightPos[2]))

    sceneContent = sceneContent.replace("LOOKAT",  str(scene.lookAt[0])+","+str(scene.lookAt[1])+","+str(scene.lookAt[2]))


    objsText = ""

    if scene.enSphere:
        objsText += """
sphere {
  <0.0, 0.0, -2.8>, 0.03
  texture {
    pigment { color Yellow }
  }
}"""

    if scene.enBox:
        boxcenter = [-0.1, 0.0, -2.8]
        boxextend = [0.08, 0.08, 0.08]

        # positions of edges of box as string
        #edgePointsAsStr = "<-0.7, -0.05, -2.9>, <-0.5, 0.05, -2.7>"

        edgePointsAsStr = f"<{boxcenter[0]-boxextend[0]/2.0},{boxcenter[1]-boxextend[1]/2.0},{boxcenter[2]-boxextend[2]/2.0}>, <{boxcenter[0]+boxextend[0]/2.0},{boxcenter[1]+boxextend[1]/2.0},{boxcenter[2]+boxextend[2]/2.0}>"
        
        objsText += """
box {
  POSS
  texture {
    pigment { color Yellow }
  }
}
""".replace("POSS", edgePointsAsStr) # replace position with src-code of position


    sceneContent = sceneContent.replace("OBJS", objsText)


    f = open("TEMPScene.pov", 'w')
    f.write(sceneContent)
    f.close()

    #import os
    #os.system("povray TEMPScene.pov"+" +W"+str(displaySize[0])+" +H"+str(displaySize[1]))

    import subprocess
    subprocess.call(["povray", "TEMPScene.pov", "+W"+str(displaySize[0]), "+H"+str(displaySize[1])], stderr=subprocess.PIPE)

import math
import cv2
from Trainer1 import * # import UL classifier

def main():
    import pygame

    c = C() # UL classifier

    t = 0.0 # time

    scene = Scene() # create scene


    pygame.init()


    displaySize = (400, 300)


    gameDisplay = pygame.display.set_mode(displaySize)
    pygame.display.set_caption('ENV')
    
    
    
    lastFrameGray = None # we don't have any last frame

    while True: # "game"loop
        t+=0.3 # advance time

        scene.cameraPos = [0.0, 0.2, -3.0]
        scene.lookAt = [0.0, 0.2-1.0, -3.0+1.0]


        # jiggle around light so that stimulus isn't exactly equal
        scene.lightPos = [0.0+((t*0.1) % 2.0) + math.cos(t*50.0)*110.0, 2.0, -3.0]
        
        # iterate over the list of Event objects 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # deactivates the pygame library 
                pygame.quit() 
  
                # quit the program. 
                quit() 

        black = (0, 0, 0)
        gameDisplay.fill(black) # fill background

        """
        # render scene to compute BB of object A
        if True: # block
            scene.enBox = True
            scene.enSphere = False

            renderScene(scene, displaySize) # render scene with renderer

            # compute bounding box of object by synthetic mask
            #
            # we use a synthetic mask because we want to focus effort on UL with prototypes,
            # not generation of proposal BB from natural images
            #
            #
            # see https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
            img = cv2.imread("TEMPScene.png")
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, imgBinary = cv2.threshold(imgGray,12,255,cv2.THRESH_BINARY)
            synMaskRect = cv2.boundingRect(imgBinary)
            del imgBinary # remove to avoid confusion
            del img
            del imgGray
            del ret
        """


        
        scene.enBox = True
        scene.enSphere = True



        renderScene(scene, displaySize) # render scene with renderer

        image = pygame.image.load("TEMPScene.png")     
        gameDisplay.blit(image, (0, 0))


        img = cv2.imread("TEMPScene.png")
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if lastFrameGray is None:
            lastFrameGray = imgGray.copy() # we need to init frame

        # compute optical flow
        # commented because not yet used
        #flow = cv2.calcOpticalFlowFarneback(lastFrameGray,imgGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # commented because it doesn't work
        #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        #hsv = np.zeros_like(imgGray)
        #hsv[...,1] = 255
        #hsv[...,0] = ang*180/np.pi/2
        #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        #flowRgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        
        # compute motion, binarize motion, segment motion
        diffImgGray = cv2.absdiff(imgGray, lastFrameGray) # compute difference between current image and last image
        ret, diffImgBinary = cv2.threshold(diffImgGray,12,255,cv2.THRESH_BINARY) # threshold to get mask of regions which moved enough
        
        kernel = np.ones((3,3),np.uint8)
        diffImgBinary = cv2.dilate(diffImgBinary,kernel,iterations = 7)
        diffImgBinary = cv2.erode(diffImgBinary,kernel,iterations = 7)

        # * segment motion into regions
        

        contours, hierachy = cv2.findContours(diffImgBinary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        del hierachy
        for iContour in contours:
            synMaskRect = cv2.boundingRect(iContour) # compute bounding box of object

            # draw bounding lines
            pygame.draw.rect(gameDisplay, (255,0,0,127), (synMaskRect[0], synMaskRect[1], synMaskRect[2], 3), width=0, border_radius=0)
            pygame.draw.rect(gameDisplay, (255,0,0,127), (synMaskRect[0], synMaskRect[1]+synMaskRect[3], synMaskRect[2], 3), width=0, border_radius=0)

            # compute BB with biggest extend
            x, y, w, h = synMaskRect
            cx, cy = x + int(w/2), y + int(h/2)
            maxExtend = max(w, h)
            x2, y2 = cx - int(maxExtend/2), cy - int(maxExtend/2)
            w2, h2 = maxExtend, maxExtend

            if w2 > 0 and h2 > 0:
                # crop by BB
                croppedImg = imgGray[y2:y2+h2, x2:x2+w2] # idx with [y:y+h, x:x+w]

                # scale cropped image to 32x32
                print(croppedImg.shape)
                rescaledCroppedImg = cv2.resize(croppedImg, (32, 32))
                print(rescaledCroppedImg.shape)

                # feed image to UL-Classifier to classify
                flattenArray = rescaledCroppedImg.flatten()
                input0 = torch.FloatTensor(flattenArray)
                sim, id = c.perceive(input0)
                print("H CLASSIFIER "+str((sim, id)))

        
        # DEBUG (image is upside down and flipped but this is fine)
        if False:
            dbgImgSurf = pygame.surfarray.make_surface(diffImgBinary)
            gameDisplay.blit(dbgImgSurf, (0, 0))
            del dbgImgSurf


        # POSTFRAME WORK
        lastFrameGray = imgGray.copy()
        
        

        # give UL classifier compute to learn
        for it in range(50):
            c.trainRound()
        
        c.decay()

        # UI
        # Draws the surface object to the screen
        pygame.display.update()

main()
