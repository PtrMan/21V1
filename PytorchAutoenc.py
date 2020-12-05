import torch.optim as optim

import torch
import torch.nn as nn

# adapted from https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
class AE(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=input_shape, out_features=128)
        self.encoder_output_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_hidden_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_output_layer = nn.Linear(in_features=128, out_features=input_shape)

    # returns reconstructed and encoded vector
    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed, code

def run():
    model = AE(32*32)

    # list of tuple of input without target
    stimulus = []

    for i in range(15):
        input0Arr = torch.randn(1, 32*32)

        #input0Arr = [-1.0] * (32*32)
        #input0Arr[i] = 1.0
        input0 = torch.FloatTensor(input0Arr)

        stimulus.append((input0))

    criterion = nn.MSELoss()
    # create optimizer
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(0, 150):
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

#run() # run experiment when this module is run