from PIL import Image
import numpy as np
import torch

def loadImageAsGrayscale(path):
    img = Image.open(path)
    img.load()
    img = img.convert('L') # convert to grayscale
    data = np.asarray(img, dtype="int32")
    return list(map(lambda v: float(v)/255.0, data.flatten())) # we want flattened array

# is supposed to train autoencoder with automatically generated images



# list of tuple of input without target
stimulus = []

if True:
    input0Arr = loadImageAsGrayscale("Scene0.png")
    input0 = torch.FloatTensor(input0Arr)
    stimulus.append((input0))

if True:
    input0Arr = loadImageAsGrayscale("Scene1.png")
    input0 = torch.FloatTensor(input0Arr)
    stimulus.append((input0))



from PytorchAutoenc import *

model = AE(32*32)

criterion = nn.MSELoss()
# create optimizer
#optimizer = optim.SGD(model.parameters(), lr=0.01)
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for i in range(0, 100):
    for iSampleInput in stimulus:
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        out, hidden = model(iSampleInput)
        #print(out)
        lossTensor = criterion(out, iSampleInput) # returns a tensor
        
        #print(loss)
        lossTensor.backward()

        # perform parameter update based on current gradients
        optimizer.step()
    
    # TODO< print only loss >
    print(lossTensor.item())

    optimizer.step()    # Does the update


# TODO< add edge detection with multiple channels >




# load stimulus scene
input0Arr = loadImageAsGrayscale("SceneStimulus0.png")
stimulusTest0 = torch.FloatTensor(input0Arr)



from scipy import spatial

# TODO< compute similarity to prototypes with cosine similarity >

# compute similarity to all prototypes
bestSim, bestIdx = 10000.0, -1
iIdx = 0
for iPrototype in stimulus:
    out, hiddenA = model(stimulusTest0)
    out, hiddenB = model(iPrototype)


    #print(hiddenA) # DEBUG
    cosineDist = spatial.distance.cosine(hiddenA.detach().numpy(), hiddenB.detach().numpy())
    if cosineDist < bestSim:
        bestSim, bestIdx = cosineDist, iIdx

    print(cosineDist)
    iIdx+=1

print((bestSim, bestIdx))
