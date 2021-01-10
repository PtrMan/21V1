# program to train optical flow based proposal-NN


import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

from RenderScene import *

from Net import *


# render the scene and return grayscale image with size of 64x64
# /param copyImageDest name of image where the scene should be stored to, only for (visual) debugging
def renderSceneAndReturnImageGray64(scene, copyImageDest = None):
    # render and read training images
    renderScene(scene, (64, 64))

    if not(copyImageDest is None):
        from shutil import copyfile
        copyfile("TEMPScene.png", copyImageDest) # copy image

    img = cv2.imread("TEMPScene.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    del img
    imgGray = cv2.resize(imgGray, (64, 64))
    return imgGray

def main():
    ###########################
    ### build trainingset
    ###########################
    print(f'H build trainingset...')

    
    # list of tuple of input and target
    inputAndTarget = []
    

    # training for moving object in center
    idx = -1
    for iSceneDescMotion, iSceneDescriptionOutArr, lightBefore, lightAfter in [
        ([0.0,0.0,0.0], [0.1, 0.9], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]), # no motion 
        ([0.0,0.0,0.02], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),  # slight motion forward
        ([0.0,0.0,-0.02], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),  # slight motion backward
        ([0.0,0.0,0.06], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),  # motion forward
        ([0.0,0.0,-0.06], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),  # motion backward

        ([0.0,0.02,0.0], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),
        ([0.0,0.03,-0.02], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),
        ([0.0,-0.05,0.06], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),
        ([0.0,-0.08,-0.06], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),

        ([0.02,0.02,0.0], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),
        ([0.02,0.03,-0.02], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),
        ([0.02,-0.05,0.06], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),
        ([0.02,-0.08,-0.06], [0.9, 0.1], [0.0, 0.0, -3.0], [0.0, 0.0, -3.0]),
        ]:
        idx+=1

        scene = Scene()
        scene.cameraPos = [0.0, 0.2, -3.0]
        scene.lookAt = [0.0, 0.2-1.0, -3.0+1.0]

        scene.boxCenters[0] = [-0.0, 0.0, -2.8]
        scene.sphereCenters = []

        scene.lightPos = lightBefore

        # render and read training images
        imgBeforeGray = renderSceneAndReturnImageGray64(scene, f'trainMotion{idx}A.png')

        # move camera to get movement vector
        scene.cameraPos[0] += iSceneDescMotion[0]
        scene.cameraPos[1] += iSceneDescMotion[1]
        scene.cameraPos[2] += iSceneDescMotion[2]

        imgCurrentGray = renderSceneAndReturnImageGray64(scene, f'trainMotion{idx}B.png')

        flow = cv2.calcOpticalFlowFarneback(imgBeforeGray,imgCurrentGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flowArr = flow.flatten() # convert image array to flat array
        
        inputArr = flowArr[:].tolist()

        # [0] is object in center?
        # [1] was no object detected?
        expectedOut = iSceneDescriptionOutArr # expected output array

        # add to training set
        inputAndTarget.append((torch.tensor(inputArr), torch.tensor(expectedOut))) # add to trainingset
        del inputArr
        del expectedOut

    # training to ignore changing lighting conditions
    sceneConfigs = []
    # for box
    sceneConfigs.append({"boxesA":[[-0.0, 0.0, -2.8]], "spheresA":[],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 1.0, -3.0], "lightB":[0.0, 0.0, -3.0]})# moving light 
    sceneConfigs.append({"boxesA":[[-0.0, 0.0, -2.8]], "spheresA":[],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 0.0, -3.0], "lightB":[0.0, 1.0, -3.0]})# moving light 

    # for sphere
    sceneConfigs.append({"boxesA":[], "spheresA":[[-0.0, 0.0, -2.8]],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 1.0, -3.0], "lightB":[0.0, 0.0, -3.0]})# moving light 
    sceneConfigs.append({"boxesA":[], "spheresA":[[-0.0, 0.0, -2.8]],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 0.0, -3.0], "lightB":[0.0, 1.0, -3.0]})# moving light 



    idx = -1
    for iSceneConfig in sceneConfigs:
        idx+=1

        scene = Scene()
        scene.cameraPos = iSceneConfig["cameraPosA"]
        scene.lookAt = iSceneConfig["lookAtA"]

        scene.boxCenters = iSceneConfig["boxesA"]
        scene.sphereCenters = iSceneConfig["spheresA"]

        scene.lightPos = iSceneConfig["lightA"]

        # render and read training images
        imgBeforeGray = renderSceneAndReturnImageGray64(scene, f'trainLighting{idx}A.png')

        # move camera to get movement vector
        scene.cameraPos = iSceneConfig["cameraPosB"]

        scene.lightPos = iSceneConfig["lightB"]

        imgCurrentGray = renderSceneAndReturnImageGray64(scene, f'trainLighting{idx}B.png')

        flow = cv2.calcOpticalFlowFarneback(imgBeforeGray,imgCurrentGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flowArr = flow.flatten() # convert image array to flat array
        
        inputArr = flowArr[:].tolist()

        # [0] is object in center?
        # [1] was no object detected?
        expectedOut = iSceneDescriptionOutArr # expected output array

        # add to training set
        inputAndTarget.append((torch.tensor(inputArr), torch.tensor(expectedOut))) # add to trainingset
        del inputArr
        del expectedOut






    import pygame

    pygame.init()


    displaySize = (400, 300)


    gameDisplay = pygame.display.set_mode(displaySize)
    pygame.display.set_caption('ProposalTraining')
    
    print(f'H setup NN...')

    net = Net(64*64*2, 2)


    criterion = nn.MSELoss()
    # create optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.000001) # lr=0.01


    itCnt = -1 # iteration counter
    
    print(f'H training ...')

    while True: # "game"loop
        itCnt+=1

        # TODO< add code for supervised NN and training and storage of NN params! >

        if True: # codeblock
            optimizer.zero_grad()   # zero the gradient buffers

            for iSampleInput, iSampleTarget in inputAndTarget:
                out = net(iSampleInput)
                #print(out)
                lossTensor = criterion(out, iSampleTarget) # returns a tensor
                
                #print(loss)
                lossTensor.backward()
            
            loss = lossTensor.item()
            if loss < 1.0e-4:
                break # break because loss is small enough
            
            # print loss
            if itCnt % 10 == 0:
                print(f'L {lossTensor.item()}')
            
            optimizer.step()    # Does the update

        # store NN weights
        if itCnt % 150 == 0: # should we export?
            print(f'H store weights...')
            torch.save(net, "proposalGen.pytorch-model") # save network
            print(f'H   (done)')




        black = (0, 0, 0)
        gameDisplay.fill(black) # fill background

        # DEBUG (image is upside down and flipped but this is fine)
        if True:
            dbgImgSurf = pygame.surfarray.make_surface(imgCurrentGray)
            gameDisplay.blit(dbgImgSurf, (0, 0))
            del dbgImgSurf

        # UI
        # Draws the surface object to the screen
        pygame.display.update()

main()
