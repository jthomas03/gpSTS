# -*- coding: utf-8 -*-
#
# John C. Thomas 2022 gpSTS - tutorial version

import torch 
import torch.nn as nn
import torch.utils.data as dataloader
import torchvision
from torchvision.datasets import DatasetFolder
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import Config
import Config as conf
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_metrics(loc, labels, predictions, num_classes, epoch, epochs, metrics):
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues,ax=ax)
    ax.set_title('Confusion matrix, $N_{epoch}$ = '+str(epoch))
    ax.set_ylabel('Actual label')
    ax.set_xlabel('Predicted label')
    ax.autoscale()
    ax = fig.add_subplot(1, 3, 2)
    name = 'Loss'
    ax.plot(epochs, metrics['loss'], label='Train', color='blue')
    ax.plot(epochs, metrics['val_'+'loss'], linestyle="--", label='Val', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(name)
    ax.set_title('1st Layer: 128 channels, 2nd Layer: 256 channels')
    ax.autoscale()
    ax.legend()
    ax = fig.add_subplot(1, 3, 3)
    name = 'Accuracy'
    ax.plot(epochs, metrics['accuracy'], label='Train', color='blue')
    ax.plot(epochs, metrics['val_'+'accuracy'], linestyle="--", label='Val', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(name)
    ax.autoscale()
    ax.legend()
    plt.savefig(loc+'metrics_'+str(epoch)+'.png', bbox_inches='tight')
    plt.close()

def plot_cm(loc, epoch, labels, predictions, p=0.5, num_classes=2):
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues)
    plt.title('Confusion matrix, $N_{epoch}$ = '+str(epoch))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.autoscale()
    plt.gca().set_aspect('equal')
    plt.savefig(loc+'cm_'+str(epoch)+'.png', bbox_inches='tight')
    plt.close()

def plot_metrics2(loc, epoch, epochs, metrics):
    plt.figure(figsize=(16, 12))
    for n, metric in enumerate(['loss', 'accuracy']):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,3,n+1)
        plt.plot(epochs, metrics[metric], label='Train', color='blue')
        plt.plot(epochs, metrics['val_'+metric], linestyle="--", label='Val', color='red')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.autoscale()
        plt.legend()
    plt.autoscale()
    plt.legend()
    plt.savefig(loc+'metrics_'+str(epoch)+'.png', bbox_inches='tight')
    plt.close()

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

# Convolutional neural network 64,128
class Conv1d(nn.Module):
    def __init__(self,num_classes=4):
        super(Conv1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
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
        size = self.size_postopt(torch.rand(1,1,conf.gpsts_config['Experiment_Settings']['NumSpectralPoints'])) # image size: 64x32
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