# run with
#   python Trainer1.py 
#   (under conda environment)

from PIL import Image
import numpy as np
import torch

def loadImageAsGrayscale(path):
    img = Image.open(path)
    img.load()
    img = img.convert('L') # convert to grayscale
    data = np.asarray(img, dtype="int32")
    return list(map(lambda v: float(v)/255.0, data.flatten())) # we want flattened array



from PytorchAutoenc import *
from scipy import spatial
import random

class Prototype(object):
    def __init__(self, arr, id):
        self.arr = arr
        self.rating = 1.0 # rating which is used to decide which prototypes to keep
        self.id = id # id to keep track

# classifier
class C(object):
    def __init__(self):
        self.prototypes = []
        self.prototypeIdCntr = 0 # counter to keep track of prototype ids
        self.nPrototypes = 50 # number of prototypes
        #self.distThreshold = 1.0 # threshold to create new prototype
        self.hcosThreshold = 0.01 #0.0032 #0.03 # theshold for hidden layer measured by cosine distance to create new prototype

        self.model = AE(32*32)

        # cache optimizer
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        # Adam optimizer with learning rate 1e-3
        self._cachedOptimizer = optim.Adam(self.model.parameters(), lr=1e-3)


    
    def perceive(self, arr):
        """
        def calcDist(arr):
            bestDist, bestIdx = float('inf'), -1
            
            iIdx = 0
            for iProto in self.prototypes:
                dist = spatial.distance.euclidean(iProto.arr.detach().numpy(), arr.detach().numpy())
                if dist < bestDist:
                    bestDist, bestIdx = dist, iIdx
                iIdx+=1
            
            return bestDist, bestIdx

        # decide if we want to store new prototype based on similarity
        dist, _ = calcDist(arr)
        print("dist="+str(dist))
        store = dist > self.distThreshold # do we want to store prototype?

        if store:
            self.prototypes.append(Prototype(arr, self.prototypeIdCntr))
            self.prototypeIdCntr+=1

            # keep under AIKR
            sorted(self.prototypes, key=lambda iProto: iProto.rating)
            self.prototypes = self.prototypes[:self.nPrototypes]
        """

        #############
        # classify

        # compute similarity to all prototypes
        bestSim, bestIdx = float('inf'), -1
        iIdx = 0
        for iPrototype in self.prototypes:
            out, hiddenA = self.model(arr)
            out, hiddenB = self.model(iPrototype.arr)


            #print(hiddenA) # DEBUG
            cosineDist = spatial.distance.cosine(hiddenA.detach().numpy(), hiddenB.detach().numpy())
            if cosineDist < bestSim:
                bestSim, bestIdx = cosineDist, iIdx

            print(cosineDist)
            iIdx+=1
        
        if bestSim > self.hcosThreshold: # do we need to create new prototype?
            self.prototypes.append(Prototype(arr, self.prototypeIdCntr))
            resId = self.prototypeIdCntr# store result id
            self.prototypeIdCntr+=1

            # keep under AIKR
            sorted(self.prototypes, key=lambda iProto: iProto.rating)
            self.prototypes = self.prototypes[:self.nPrototypes]

            return (0.0, resId)
        else:
            # reward winner
            self.prototypes[bestIdx].rating = 1.0

            return (bestSim, self.prototypes[bestIdx].id)

    # training round, should be relativly fast
    def trainRound(self):
        if len(self.prototypes) == 0:
            return
        
        selIdx = random.randint(0, len(self.prototypes)-1)
        selPrototype = self.prototypes[selIdx]

        criterion = nn.MSELoss()

        # actual training
        if True:
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            self._cachedOptimizer.zero_grad()

            out, hidden = self.model(selPrototype.arr)
            #print(out)
            lossTensor = criterion(out, selPrototype.arr) # returns a tensor
            
            #print(loss)
            lossTensor.backward()

            # perform parameter update based on current gradients
            self._cachedOptimizer.step()
        
        #print("loss="+str(lossTensor.item())) # print loss

    # decay all ratings so old unused tend to die off
    def decay(self):
        for iProto in self.prototypes:
            iProto.rating *= 0.96

if __name__ == "__main__":
    c = C()

    if True:
        input0Arr = loadImageAsGrayscale("Scene0.png")
        input0 = torch.FloatTensor(input0Arr)
        sim, id = c.perceive(input0)

    # give time to learn
    for it in range(50):
        c.trainRound()

    if True:
        input0Arr = loadImageAsGrayscale("Scene1.png")
        input0 = torch.FloatTensor(input0Arr)
        sim, id = c.perceive(input0)

    # give time to learn
    for it in range(50):
        c.trainRound()





    # load stimulus scene
    input0Arr = loadImageAsGrayscale("SceneStimulus0.png")
    stimulusTest0 = torch.FloatTensor(input0Arr)
    sim, id = c.perceive(stimulusTest0)
    print((sim, id))


    # TODO LOW< add edge detection with multiple channels >

    # TODO< decay importance >
    # TODO< add to importance when it was used >
