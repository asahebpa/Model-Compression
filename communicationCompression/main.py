# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import copy
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from utils import *
import os
import os.path
from os import path
from hadautils import *
from Procedures import *
saveDir = './results/'
os.path.isdir(saveDir) or os.mkdir(saveDir)
datasetDir = './datas/'
os.path.isdir(datasetDir) or os.mkdir(datasetDir)

hiddenDim = 512
layerDimension = [hiddenDim, 1000, hiddenDim, 2000, hiddenDim] # layer dimension for each layer
activationFunction = relu # None/sigmoid/relu # activation function for random projection
nEpochs = 3
batchSizeTrain = 64
batchSizeTest = 1000
learningRate = 0.01
momentum = 0.5
logInterval = 10
seed = 1010
dataset = 'FashionMNIST' # MNIST/CIFAR10/FashionMNIST
# parameters for random noise generation
mu = 0 # mean
sigma = 0.1 # standard deviation

numberLayers = len(layerDimension) # number of layer for random projection
setSeed(seed)

# cuda
cudaFlag = torch.cuda.is_available()
transformFunction = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#TODO: add onehot encoding to torchvision transforms

# Load dataset
if dataset == 'MNIST':
    # already normalized between 0 and 1 after using trainLoader, .data is not normalized
    datasetTrain = torchvision.datasets.MNIST(datasetDir, train=True, download=True,
                         transform=transformFunction)
    datasetTest = torchvision.datasets.MNIST(datasetDir, train=False, download=True,
                        transform=transformFunction)
    imgHeight, imgWidth = datasetTrain.data.shape[1:]
    numChannels = 1
    numClasses = 10
elif dataset == 'CIFAR10':
    #TODO: check normalization
    datasetTrain = torchvision.datasets.CIFAR10(datasetDir, train=True, download=True,
                        transform=transformFunction)
    datasetTest = torchvision.datasets.CIFAR10(datasetDir, train=False, download=True,
                        transform=transformFunction)
    imgHeight, imgWidth, numChannels = datasetTrain.data.shape[1:]
    numClasses = 10
elif dataset == 'FashionMNIST':
    #TODO: check normalization
    datasetTrain = torchvision.datasets.FashionMNIST(datasetDir, train=True, download=True,
                        transform=transformFunction)
    datasetTest = torchvision.datasets.FashionMNIST(datasetDir, train=False, download=True,
                        transform=transformFunction)
    imgHeight, imgWidth = datasetTrain.data.shape[1:]
    numChannels = 1
    numClasses = 10

trainLoader = torch.utils.data.DataLoader(datasetTrain,batch_size=batchSizeTrain, shuffle=True)
testLoader = torch.utils.data.DataLoader(datasetTest,batch_size=batchSizeTest, shuffle=True)

# Network Architectures
class Net(nn.Module):
    def __init__(self, dim=None, activationFunction=None, batchNorm=False):
        super().__init__()
        self.actFunc = activationFunction
        self.inputDim = imgHeight * imgWidth * numChannels
        self.hiddenDim = hiddenDim if dim is None else int((dim - self.inputDim) / (
                    self.inputDim + numClasses + 1)) + 1  # fix number of params by approximating the hidden dimension

        self.fc1 = nn.Linear(self.inputDim, self.hiddenDim)
        self.fc2 = nn.Linear(self.hiddenDim, numClasses)
        self.batchNorm = batchNorm
        if self.batchNorm:
            self.bn = nn.BatchNorm1d(self.hiddenDim)
    def forward(self, x):
        x = x.reshape(-1, imgHeight * imgWidth * numChannels)

        x = self.fc1(x)

        if self.actFunc != None:
            x = self.actFunc(x)

        if self.batchNorm:
            x = self.bn(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def train(network, epoch):
    network.train()
    if cudaFlag:
        network = network.cuda()
    for batchIdx, (data, target) in enumerate(trainLoader):
        optimizer.zero_grad()
        # move to cuda
        if cudaFlag:
            target = target.cuda()
            data = data.cuda()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(network.parameters(), 15)
        optimizer.step()

def test(network):
    network.eval()
    testLoss = 0
    correct = 0
    if cudaFlag:
        network = network.cuda()
    with torch.no_grad():
        for data, target in testLoader:
            if cudaFlag:
                target = target.cuda()
                data = data.cuda()
            output = network(data)
            testLoss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    testLoss /= len(testLoader.dataset)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        testLoss, correct, len(testLoader.dataset),
        100. * correct / len(testLoader.dataset)))
    return correct / len(testLoader.dataset)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(saveDir+dataset+"state_dict_model.pt")
    print(path.isfile(saveDir+dataset+"state_dict_model.pt"))
    print('Dataset is ', dataset)
    print('-' * 100)
    if path.isfile(saveDir+dataset+"state_dict_model.pt") == False:
        network = Net(batchNorm=True, activationFunction=activationFunction)
        optimizer = optim.SGD(network.parameters(), lr=learningRate, momentum=momentum)
        # number of parameters

        print('#INFO: --> Number of Params #', count_parameters(network))
        # print('#INFO: Network Configuration is ',network)
        test(network)
        for epoch in range(1, nEpochs + 1):
            train(network, epoch)
            test(network)
        torch.save(network.state_dict(), saveDir+dataset+"state_dict_model.pt")
        del network
    else:
        print('Loading Model:')
        whole_flags = [False, False]
        Levels = [0.95, 0.95]
        compressdim = [0, 1]
        network = Net(batchNorm=True, activationFunction=activationFunction)
        network.load_state_dict(torch.load(saveDir + dataset + "state_dict_model.pt"))
        network2 = network
        test(network)
        paramorgtot = []
        paramcompresstot = []
        Lay = 0
        ptemp = np.asarray(network.fc1.weight.detach().cpu())
        Decompressedp, paramorg, paramcompress = Compression_procedure(ptemp, compressdim[Lay],Levels[Lay], whole_flags[Lay],1)
        paramorgtot.append(paramorg[0])
        paramcompresstot.append(paramcompress[0])
        network2.fc1.weight = torch.nn.Parameter(data=torch.Tensor(Decompressedp), requires_grad=True)

        Lay = 1
        ptemp = np.asarray(network.fc2.weight.detach().cpu())
        Decompressedp, paramorg, paramcompress = Compression_procedure(ptemp, compressdim[Lay], Levels[Lay], whole_flags[Lay],1)
        paramorgtot.append((paramorg[0]))
        paramcompresstot.append((paramcompress[0]))
        network2.fc2.weight = torch.nn.Parameter(data=torch.Tensor(Decompressedp), requires_grad=True)
###########################################
        cor = test(network2)
        comprate = np.sum(np.asarray(paramcompresstot))/np.sum(np.asarray(paramorgtot))
        print(comprate)
        print(np.sum(np.asarray(paramcompresstot)), np.sum(np.asarray(paramorgtot)))



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
