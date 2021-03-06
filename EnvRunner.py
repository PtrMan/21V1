# supposed to simulate primitive robot in a virtual 2.5 environment

import math
import cv2
import torch
import torch.nn as nn

from Trainer1 import * # import UL classifier

from RenderScene import *
from Net import * # import feed forward network for proposal generator

from NetUtils import * # functions for network

class ObjectProposal(object):
    def __init__(self, cx, cy, size):
        self.cx = cx
        self.cy = cy
        self.size = size

def main():
    net = torch.load("proposalGen.pytorch-model") # load proposal generator network

    import pygame

    c = C() # UL classifier

    t = 0.0 # time

    scene = Scene() # create scene
    scene.objs.append(Obj("s", [0.0, 0.0, -2.8], 0.03)) # default sphere


    pygame.init()


    displaySize = (400, 300)


    gameDisplay = pygame.display.set_mode(displaySize)
    pygame.display.set_caption('ENV')
    
    
    
    lastFrameGray = None # we don't have any last frame

    while True: # "game"loop
        t+=0.3 # advance time

        scene.cameraPos = [0.0, 0.2, -3.0]
        scene.lookAt = [0.0, 0.2-1.0, -3.0+1.0]

        scene.objs[0].center = [math.cos(t)*0.02, math.sin(t)*0.02, -2.8]

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
        flow = cv2.calcOpticalFlowFarneback(lastFrameGray,imgGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # compute object proposals by motion
        # * prepare
        objectProposals = [] # all object proposals by motion

        proposalSize = int(128*1.5)# imagesize used for proposal generator

        height, width = flow.shape[:2]
        nX = int(width / (proposalSize/3/2)) # count of x iterations
        nY = int(height / (proposalSize/3/2)) # count of y iterations
        # * actual computation
        for ix in range(nX):
            for iy in range(nY):
                

                # calc center position
                cx = int(proposalSize/2 + proposalSize/3/2 * ix)
                cy = int(proposalSize/2 + proposalSize/3/2 * iy)

                # crop motion image
                croppedMotionImg = flow[cy-int(proposalSize/2):cy+int(proposalSize/2), cx-int(proposalSize/2):cx+int(proposalSize/2)] # idx with [y:y+h, x:x+w]

                # scale cropped image to fied size to feed into NN
                croppedMotionImg = cv2.resize(croppedMotionImg, (64, 64))

                # feed into NN
                flowArr = croppedMotionImg.flatten().tolist() # convert image array to flat array
                out = net(torch.tensor(flowArr))

                # interpret result
                print(f'{out[0]} {out[1]}')

                cls_ = calcClass(out)
                if cls_ == 0:
                    # was selection criterion [0] chosen?
                    # add to proposals because this was selected as a proposal
                    objectProposals.append(ObjectProposal(cx,cy,int(proposalSize/3))) # proposal is only in the center, that's why it's 1.0/3.0
                elif cls_ == n-1: # case when the proposal wasn't made for this cropped image
                    pass


                # for debugging grid where proposals are sampled
                #objectProposals.append(ObjectProposal(cx,cy,int(proposalSize/3))) # proposal is only in the center, that's why it's 1.0/3.0
                

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

        # draw debug of bounding box
        # /param rect rect of the bounding box, ex: (5, 10, 15, 17)
        # /param color color as tuple, ex: (255,0,0,127)
        def drawBb(rect, color):
            pygame.draw.rect(gameDisplay, color, (rect[0], rect[1], rect[2], 3), width=0, border_radius=0)
            pygame.draw.rect(gameDisplay, color, (rect[0], rect[1]+rect[3], rect[2], 3), width=0, border_radius=0)


        # * segment motion into regions
        

        contours, hierachy = cv2.findContours(diffImgBinary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        del hierachy
        for iContour in contours:
            synMaskRect = cv2.boundingRect(iContour) # compute bounding box of object

            # draw bounding lines
            drawBb(synMaskRect,(255,0,0,127))

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
        
        # DEBUG draw object proposal rectanges by motion
        if True:
            for iObjectProposal in objectProposals:
                rect = (int(iObjectProposal.cx-iObjectProposal.size/2), int(iObjectProposal.cy-iObjectProposal.size/2), iObjectProposal.size, iObjectProposal.size)
                drawBb(rect, (0,0,255,127))



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
