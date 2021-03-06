# program to train optical flow based proposal-NN

import time

import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

from RenderScene import *

from Net import *


from RoboMisc import * # spongebot specific helpers



def loadImageAsImageGrayRect(imagename, size):
    img = cv2.imread(imagename)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    del img
    imgGray = cv2.resize(imgGray, (size, size))
    return imgGray

# render the scene and return grayscale image with size of 64x64
# /param scenename is the name of the scene file
# /param copyImageDest name of image where the scene should be stored to, only for (visual) debugging
def renderSceneAndReturnImageGray64(scene, scenename, copyImageDest = None):
    # render and read training images
    renderScene2(scene, (64, 64), scenename+".pov")

    if not(copyImageDest is None):
        from shutil import copyfile
        copyfile(scenename+".png", copyImageDest) # copy image

    return loadImageAsImageGrayRect(scenename+".png", 64)


import random # to gnerate a lot of training data
import math

def vecAdd(a,b):
    return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]

from NetUtils import * # functions for network

import threading

# render a scene, change it and render it again: is used to generate training data in parallel
# /param changeFn called to change the scene
def thread_0(scene, scenename0, changeFn, scenename1):
    renderSceneAndReturnImageGray64(scene, scenename0)
    changeFn(scene)
    renderSceneAndReturnImageGray64(scene, scenename1)



# training to ignore changing lighting conditions
# returns all inputAndTarget tuples of it
def genChangingLightingConditions():
    renderThreads = [] # collect all worker threads to render in parallel

    sceneConfigs = []

    random.seed(42+7)

    diffvecs = []
    for i in range(5):
        diffvecs.append([random.uniform(-1.0, 1.0),random.uniform(-1.0, 1.0),random.uniform(-1.0, 1.0)])
    
    # differences from the original position
    posdiffs = []
    for i in range(9):
        posdiffs.append([random.uniform(-1.0, 1.0)*0.18,random.uniform(-1.0, 1.0)*0.18,random.uniform(-1.0, 1.0)*0.18])
    
    for iPosDiff in posdiffs: # iterate over variation of position
        for iDiffvec in diffvecs: # iterate over difference vectors for light positions
            # for box
            sceneConfigs.append({"boxesA":[vecAdd([-0.0, 0.0, -2.8],iPosDiff)], "spheresA":[],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 1.0, -3.0], "lightB":vecAdd([0.0, 1.0, -3.0], iDiffvec)})# moving light 
            sceneConfigs.append({"boxesA":[vecAdd([-0.0, 0.0, -2.8],iPosDiff)], "spheresA":[],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 0.0, -3.0], "lightB":vecAdd([0.0, 0.0, -3.0], iDiffvec)})# moving light 

            # for sphere
            for iSphereR in [0.02, 0.03, 0.04, 0.055]: # vary sphere radius
                sceneConfigs.append({"boxesA":[], "spheresA":[(vecAdd([-0.0, 0.0, -2.8],iPosDiff), iSphereR)],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 1.0, -3.0], "lightB":vecAdd([0.0, 1.0, -3.0], iDiffvec)})# moving light 
                sceneConfigs.append({"boxesA":[], "spheresA":[(vecAdd([-0.0, 0.0, -2.8],iPosDiff), iSphereR)],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 0.0, -3.0], "lightB":vecAdd([0.0, 0.0, -3.0], iDiffvec)})# moving light 



    idx = -1
    for iSceneConfig in sceneConfigs:
        iSceneDescriptionOutArr = [0.1,0.9] # no motion -> no proposal
        
        idx+=1

        scene = Scene()
        scene.backgroundColor = [0.0, 1.0, 0.0] # green for better visualization
        scene.cameraPos = iSceneConfig["cameraPosA"]
        scene.lookAt = iSceneConfig["lookAtA"]

        for iBoxCenter in iSceneConfig["boxesA"]:
            scene.objs.append(Obj("b", iBoxCenter, [0.08, 0.08, 0.08]))
        for iSpherePos, iSphereR in iSceneConfig["spheresA"]:
            scene.objs.append(Obj("s", iSpherePos, iSphereR))

        scene.lightPos = iSceneConfig["lightA"]


        # is called to change the scene in the worker thread
        def changeScene(scene):
            # move camera to get movement vector
            scene.cameraPos = iSceneConfig["cameraPosB"]

            scene.lightPos = iSceneConfig["lightB"]
        
        renderThread = threading.Thread(target=thread_0, args=(scene, f'trainLighting{idx}A', changeScene, f'trainLighting{idx}B'), daemon=True)
        renderThread.start()
        renderThreads.append(renderThread)

    # wait for completion of all render threads
    print("H waiting for completion of renderthreads...")
    iIdx = 0
    for iRenderThread in renderThreads:
        print(f'H ... {iIdx}')
        iRenderThread.join()
        
        iIdx += 1

    print("H compute motion for images ...")


    inputAndTarget = []

    idx = -1
    for iSceneConfig in sceneConfigs:
        iSceneDescriptionOutArr = [0.1,0.9] # no motion -> no proposal
        
        idx+=1

        imgBeforeGray = loadImageAsImageGrayRect(f'trainLighting{idx}A.png', 64)
        imgCurrentGray = loadImageAsImageGrayRect(f'trainLighting{idx}B.png', 64)


        flow = cv2.calcOpticalFlowFarneback(imgBeforeGray,imgCurrentGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flowArr = flow.flatten() # convert image array to flat array
        
        inputArr = flowArr[:].tolist()

        # [0] is object in center?
        # [1] was no object detected?
        expectedOut = iSceneDescriptionOutArr # expected output array

        # add to training set
        inputAndTarget.append((torch.tensor(inputArr), torch.tensor(expectedOut), "ignoreLightA")) # add to trainingset
        del inputArr
        del expectedOut
    
    return inputAndTarget



def main():
    ###########################
    ### build trainingset
    ###########################
    print(f'H build trainingset and testset...')

    
    inputAndTarget = [] # trainingset as tuple (input, target, type), where type is string

    inputAndTargetTest = [] # testset as tuple (input, target, type), where type is string
    
    

    # TODO REFACTOR< append result to array >
    for iInputAndTarget in genChangingLightingConditions():
        inputAndTarget.append(iInputAndTarget)



    if False:# training to ignore motion of object which is not in center
        sceneConfigs = []

        random.seed(42+7)

        diffvecs = []
        for i in range(35):
            diffvecs.append([random.uniform(-1.0, 1.0)*0.07,random.uniform(-1.0, 1.0)*0.07,random.uniform(-1.0, 1.0)*0.07])
        
        # differences from the center
        centerdiffs = []
        for i in range(14*2*2):
            centerdiffs.append([random.uniform(-1.0, 1.0)*0.22,random.uniform(-1.0, 1.0)*0.22,random.uniform(-1.0, 1.0)*0.22])
        
        for iCenterDiff in centerdiffs: # iterate over variation of position
            for iDiffvec in diffvecs: # iterate over difference vectors for light positions
                # for box
                #sceneConfigs.append({"boxesA":[vecAdd([-0.0, 0.0, -2.8],iPosDiff)], "spheresA":[],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 1.0, -3.0], "lightB":vecAdd([0.0, 1.0, -3.0], iDiffvec)})# moving light 
                #sceneConfigs.append({"boxesA":[vecAdd([-0.0, 0.0, -2.8],iPosDiff)], "spheresA":[],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 0.0, -3.0], "lightB":vecAdd([0.0, 0.0, -3.0], iDiffvec)})# moving light 

                # for sphere
                for iSphereR in [0.02, 0.04, 0.06, 0.09]: # vary sphere radius
                    sceneConfigs.append({"boxesA":[], "spheresA":[(vecAdd([-0.0, 0.0, -2.8],iCenterDiff), iSphereR)],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":vecAdd([0.0, 0.2, -3.0],iDiffvec), "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 1.0, -3.0], "lightB":[0.0, 1.0, -3.0]})# moving light 

        #######
        # testset
        diffvecs = []
        for i in range(5):
            diffvecs.append([random.uniform(-1.0, 1.0)*0.07,random.uniform(-1.0, 1.0)*0.07,random.uniform(-1.0, 1.0)*0.07])
        
        # differences from the center
        centerdiffs = []
        for i in range(14):
            centerdiffs.append([random.uniform(-1.0, 1.0)*0.22,random.uniform(-1.0, 1.0)*0.22,random.uniform(-1.0, 1.0)*0.22])
        
        for iCenterDiff in centerdiffs: # iterate over variation of position
            for iDiffvec in diffvecs: # iterate over difference vectors for light positions
                # for box
                #sceneConfigs.append({"boxesA":[vecAdd([-0.0, 0.0, -2.8],iPosDiff)], "spheresA":[],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 1.0, -3.0], "lightB":vecAdd([0.0, 1.0, -3.0], iDiffvec)})# moving light 
                #sceneConfigs.append({"boxesA":[vecAdd([-0.0, 0.0, -2.8],iPosDiff)], "spheresA":[],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":[0.0, 0.2, -3.0], "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 0.0, -3.0], "lightB":vecAdd([0.0, 0.0, -3.0], iDiffvec)})# moving light 

                # for sphere
                for iSphereR in [0.09]: # vary sphere radius
                    sceneConfigs.append({"boxesA":[], "spheresA":[(vecAdd([-0.0, 0.0, -2.8],iCenterDiff), iSphereR)],  "cameraPosA":[0.0, 0.2, -3.0],"lookAtA":[0.0, 0.2-1.0, -3.0+1.0], "cameraPosB":vecAdd([0.0, 0.2, -3.0],iDiffvec), "sceneDescriptionOutArr":[0.1, 0.9], "lightA":[0.0, 1.0, -3.0], "lightB":[0.0, 1.0, -3.0], "testset":True})# moving light 




        idx = -1
        for iSceneConfig in sceneConfigs:
            iSceneDescriptionOutArr = [0.1,0.9] # no motion -> no proposal
            
            idx+=1

            scene = Scene()
            scene.backgroundColor = [0.0, 1.0, 0.0] # green for better visualization
            scene.cameraPos = iSceneConfig["cameraPosA"]
            scene.lookAt = iSceneConfig["lookAtA"]

            scene.boxCenters = iSceneConfig["boxesA"]
            scene.spheres = iSceneConfig["spheresA"]

            scene.lightPos = iSceneConfig["lightA"]

            # render and read training images
            imgBeforeGray = renderSceneAndReturnImageGray64(scene, f'trainNonCenter{idx}A.png')

            # move camera to get movement vector
            scene.cameraPos = iSceneConfig["cameraPosB"]

            scene.lightPos = iSceneConfig["lightB"]

            imgCurrentGray = renderSceneAndReturnImageGray64(scene, f'trainNonCenter{idx}B.png')

            flow = cv2.calcOpticalFlowFarneback(imgBeforeGray,imgCurrentGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flowArr = flow.flatten() # convert image array to flat array
            
            inputArr = flowArr[:].tolist()

            # [0] is object in center?
            # [1] was no object detected?
            expectedOut = iSceneDescriptionOutArr # expected output array

            if "testset" in iSceneConfig:
                inputAndTargetTest.append((torch.tensor(inputArr), torch.tensor(expectedOut), "ignoreLightA")) # add to testset
            else:
                inputAndTarget.append((torch.tensor(inputArr), torch.tensor(expectedOut), "ignoreLightA")) # add to trainingset
            
            del inputArr
            del expectedOut





    if False:# training to ignore perspective parallax (for now for SpongeBot setting)

        random.seed(42+8)

        diffvecs = []
        diffvecs.append([0.0,0.0,0.0]) # no motion
        for iMagX in [0.0, 0.02, 0.05, 0.1]: # iterate over magnitude of motion
            diffvecs.append([0.0,0.0,iMagX]) # side
            diffvecs.append([0.0,0.0,-iMagX]) # side
        #    diffvecs.append([0.0,iMagX,0.0])
        #    diffvecs.append([0.0,-iMagX,0.0])
            diffvecs.append([iMagX,0.0,0.0]) # forward
            diffvecs.append([-iMagX,0.0,0.0]) # backward
        
        cameraRotDiffs = []
        cameraRotDiffs.append(0.0)
        cameraRotDiffs.append(0.05)
        cameraRotDiffs.append(-0.05)
        cameraRotDiffs.append(0.1)
        cameraRotDiffs.append(-0.1)
        cameraRotDiffs.append(0.2)
        cameraRotDiffs.append(-0.2)


        for i in range(8):
            x =random.uniform(-1.0, 1.0)
            y =random.uniform(-1.0, 1.0)
            z =random.uniform(-1.0, 1.0)

            diffvecs.append([x*0.02,0.0,z*0.02])
            diffvecs.append([x*0.06,0.0,z*0.06])

        iid = -1
        for iCamRotDiff in cameraRotDiffs:
            for iDiffvec in diffvecs:
                iid+=1

                isAnyMotion = False #parallax motion doesn't indicate objects!

                iSceneDescriptionOutArr = [0.1,0.9] # no motion -> no proposal
                if isAnyMotion:
                    iSceneDescriptionOutArr = [0.9,0.1]


                lightBefore = [0.0, 0.0, -3.0]
                lightAfter = [0.0, 0.0, -3.0]

                camDir = spongebotCalcCameraDirByAngle(0.0) # compute direction of camera

                scene = Scene()
                scene.backgroundColor = [0.0, 1.0, 0.0] # green for better visualization
                scene.cameraPos = [0.0, 0.2, -3.0]            
                scene.lookAt = vecAdd(scene.cameraPos,camDir)

                scene.boxCenters = [] # legacy
                scene.spheres = [] # legacy
                scene.enBox = False # legacy

                scene.lightPos = lightBefore

                # add floor to scene
                obj = Obj("b", [0.0, 0.0, 0.0], [50.0, 0.01, 50.0])
                obj.isTextured = True # 
                scene.objs.append(obj)
                del obj

                # render and read training images
                imgCurrentGray = renderSceneAndReturnImageGray64(scene, f'trainParallaxPerspectiveA{iid}B.png')

                
                camDir = spongebotCalcCameraDirByAngle(0.0-iCamRotDiff) # compute direction of camera

                # move camera
                scene.cameraPos[0] -= iDiffvec[0]
                scene.cameraPos[1] -= iDiffvec[1]
                scene.cameraPos[2] -= iDiffvec[2]

                scene.lookAt = vecAdd(scene.cameraPos,camDir)


                imgBeforeGray = renderSceneAndReturnImageGray64(scene, f'trainParallaxPerspectiveA{iid}A.png')

                flow = cv2.calcOpticalFlowFarneback(imgBeforeGray,imgCurrentGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flowArr = flow.flatten() # convert image array to flat array
                
                inputArr = flowArr[:].tolist()

                # [0] is object in center?
                # [1] was no object detected?
                expectedOut = iSceneDescriptionOutArr # expected output array

                # add to training set
                inputAndTarget.append((torch.tensor(inputArr), torch.tensor(expectedOut), "parallaxPerspectiveA")) # add to trainingset
                del inputArr
                del expectedOut
                del iSceneDescriptionOutArr




    if True:# training for moving object in center

        random.seed(42+7)

        diffvecs = []
        diffvecs.append([0.0,0.0,0.0]) # no motion
        for iMagX in [0.0, 0.02, 0.05, 0.1]: # iterate over magnitude of motion
            diffvecs.append([0.0,0.0,iMagX])
            diffvecs.append([0.0,0.0,-iMagX])
            diffvecs.append([0.0,iMagX,0.0])
            diffvecs.append([0.0,-iMagX,0.0])
            diffvecs.append([iMagX,0.0,0.0])
            diffvecs.append([-iMagX,0.0,0.0])


        for i in range(8):
            x =random.uniform(-1.0, 1.0)
            y =random.uniform(-1.0, 1.0)
            z =random.uniform(-1.0, 1.0)

            diffvecs.append([x*0.02,y*0.02,z*0.02])
            diffvecs.append([x*0.06,y*0.06,z*0.06])

        idx = -1
        for iDiffvec in diffvecs:
            idx+=1

            isAnyMotion = math.sqrt(iDiffvec[0]*iDiffvec[0]+ iDiffvec[1]*iDiffvec[1]+ iDiffvec[2]*iDiffvec[2]) > 0.001

            iSceneDescriptionOutArr = [0.1,0.9] # no motion -> no proposal
            if isAnyMotion:
                iSceneDescriptionOutArr = [0.9,0.1]


            lightBefore = [0.0, 0.0, -3.0]
            lightAfter = [0.0, 0.0, -3.0]

            scene = Scene()
            scene.backgroundColor = [0.0, 1.0, 0.0] # green for better visualization
            scene.cameraPos = [0.0, 0.2, -3.0]
            scene.lookAt = [0.0, 0.2-1.0, -3.0+1.0]

            for iBoxCenter in [[-0.0, 0.0, -2.8]]:
                scene.objs.append(Obj("b", iBoxCenter, [0.08, 0.08, 0.08]))

            scene.lightPos = lightBefore

            # render and read training images
            imgCurrentGray = renderSceneAndReturnImageGray64(scene, f'trainMotion{idx}B.png')

            # move camera to get movement vector
            scene.cameraPos[0] -= iDiffvec[0]
            scene.cameraPos[1] -= iDiffvec[1]
            scene.cameraPos[2] -= iDiffvec[2]

            imgBeforeGray = renderSceneAndReturnImageGray64(scene, f'trainMotion{idx}A.png')

            flow = cv2.calcOpticalFlowFarneback(imgBeforeGray,imgCurrentGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flowArr = flow.flatten() # convert image array to flat array
            
            inputArr = flowArr[:].tolist()

            # [0] is object in center?
            # [1] was no object detected?
            expectedOut = iSceneDescriptionOutArr # expected output array

            # add to training set
            inputAndTarget.append((torch.tensor(inputArr), torch.tensor(expectedOut), "motionA")) # add to trainingset
            del inputArr
            del expectedOut
            del iSceneDescriptionOutArr








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

    lossAvg = None

    trainingStartT = time.time()

    while True: # "game"loop
        itCnt+=1

        if True: # codeblock
            optimizer.zero_grad()   # zero the gradient buffers

            for ii in range(20):
                iSampleInput, iSampleTarget, type_ = random.choice(inputAndTarget)

                out = net(iSampleInput)
                #print(out)
                lossTensor = criterion(out, iSampleTarget) # returns a tensor
                
                #print(loss)
                lossTensor.backward()
            
            loss = lossTensor.item()
            
            if not(lossAvg is None):
                lossAvg = lossAvg*0.999 + loss*0.001 # average loss
            else:
                lossAvg = loss

            if lossAvg < 1.0e-8:
                break # break because loss is small enough

            # print loss
            if itCnt % 6 == 0:
                print(f'H epoch = {itCnt/len(inputAndTarget)}')
                print(f'L {lossTensor.item()}')
                print(f'H loss avg = {lossAvg}')
                print(f'H time = {time.time()-trainingStartT}')
            
            optimizer.step()    # Does the update
        


        # store NN weights
        if itCnt % 150 == 0: # should we export?
            print(f'H store weights...')
            torch.save(net, "proposalGen.pytorch-model") # save network
            print(f'H   (done)')

        # run testset
        if True: #codeblock
            if itCnt % 50 == 0:
                n = 0
                np = 0 # positive

                for iSampleInput, iSampleTarget, type_ in inputAndTargetTest:
                    n+=1

                    out = net(iSampleInput)

                    classTestset = calcClass(iSampleTarget)
                    classResult = calcClass(out)
                    
                    if classTestset == classResult: # is the classification correct?
                        np+=1
                
                print(f'H testset n={n} np={np} ratio={float(np)/float(n)}')



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


if __name__ == "__main__":
    main()
