import torch 
import torch.nn as nn
import torch.utils.data as dataloader
import torchvision
#import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import Config
import Config as conf
#import os
#import glob
#from typing import Any, Callable, cast, Dict, List, Optional, Tuple
#import time

### Data loader ###
#def predict_image(image):
#    input = torch.from_numpy(image)
#    input = input.unsqueeze(1)
#    input = input.to(device)
#    output = model(input)
#    index = output.data.cpu().numpy().argmax()
#    prob = F.softmax(output, dim=1)
#    return index

#def imrescale(im,amin=0,amax=255):
#    tmin = im.min()
#    tmax = im.max()
#    out = np.zeros([im.shape[0],im.shape[1]]).astype('uint8')
#    for i in range(0,im.shape[0]):
#        for j in range(0,im.shape[1]):
#            out[i][j] = int((((im[i][j] - tmin)*amax)/(tmax-tmin))+amin)
#    return out

#def adjust_learning_rate(epoch, lrate):
#    if epoch > 30:
#        learning_rate = learning_rate / 10
#    elif epoch > 60:
#        learning_rate = learning_rate / 100
#    elif epoch > 90:
#        learning_rate = learning_rate / 1000
#    elif epoch > 120:
#        learning_rate = learning_rate / 10000
#    elif epoch > 150:
#        learning_rate = learning_rate / 100000
#    elif epoch > 180:
#        learning_rate = learning_rate / 1000000
#    return learning_rate

def make_predictions(model, device, test_loader):
    # Set model to eval mode to notify all layers.
    model.eval()
    targets = []
    preds = []
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for sample in test_loader:
            data, target = sample
            data, target = data.to(device), target.to(device)
            # Predict for data by doing forward pass
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
    targets = [np.hstack(y) for y in targets]
    preds = [np.hstack(y) for y in preds]
    targets = np.hstack(targets)
    preds = np.hstack(preds)
    return targets, preds

def progbar(curr, total, full_progbar, epoch, num_epochs, loss, accuracy):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', 
          '#'*filled_progbar + '-'*(full_progbar-filled_progbar), 
          f'Epoch [{epoch}/{num_epochs}]', 
          f'Step [{curr}/{total}]', 
          'Loss: {:.6f}'.format(loss),
          'Accuracy: [{:>7.2%}]'.format(accuracy),
          end='')

def specnorm(data):
    dmin = np.min(data)
    dmax = np.max(data)
    out = np.zeros(data.shape[0])
    for i in range(0,data.shape[0]):
        out[i] = (data[i] - dmin)/(dmax-dmin) 
    return out

def np_loader(path):
    with open(path, 'rb') as f:
        data = np.load(f,allow_pickle=True)
        dnp = data[0]
        dnp = specnorm(dnp)
        dout = torch.from_numpy(dnp).float()
        return torch.reshape(dout,(1,len(dnp)))

def spec_loader(data):
    dnp = data
    dnp = specnorm(dnp)
    dout = torch.from_numpy(dnp).float()
    return torch.reshape(dout,(1,1,len(dnp)))

def dplot(imagein, title='Interpolated'):
    fig, ax = plt.subplots()
    z_min, z_max = imagein.min(), imagein.max()
    xx, yy = np.meshgrid(np.linspace(1, imagein.shape[0], imagein.shape[0]), np.linspace(1, imagein.shape[1], imagein.shape[1]))
    x = xx[::1]
    y = yy[::1]
    cout = ax.pcolormesh(x, y, imagein, cmap='bwr', vmin=z_min, vmax=z_max)
    ax.set_title(title)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.axis('scaled')
    fig.tight_layout()
    plt.show()

def dscplot(dx,dy,di, title='Collected Points'):
    fig = plt.figure(1, clear=True)
    plt.scatter(dx, dy, c=di, cmap='viridis') #0,1
    plt.title(title)
    plt.colorbar()
    plt.axis('scaled')
    fig.tight_layout()
    plt.show()

# Convolutional neural network
class Conv1d(nn.Module):
    def __init__(self,num_classes=4):
        super(Conv1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1))
        self.layer3 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.getinput(), num_classes)

    def size_postopt(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.size()

    def getinput(self):
        size = self.size_postopt(torch.rand(1,1,conf.nanonis_config['Nanonis_Settings']['NumSpectralPoints'])) # image size: 64x32
        m = 1
        for i in size:
            m *= i
        return int(m)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out 