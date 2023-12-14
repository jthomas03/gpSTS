# -*- coding: utf-8 -*-
#
# John C. Thomas 2022 gpSTS - tutorial version

import torch 
import torch.nn as nn
import torch.utils.data as dataloader
import torchvision
from torchvision.datasets import DatasetFolder
import random
import sys
import os
import numpy as np
import torch.nn.functional as F
import Config
import Config as conf
from gpsts.Classification.cnn1dimport import np_loader, Conv1d, progbar, make_predictions, dplot, spec_loader, dscplot, plot_metrics, plot_cm


def main_cnn():
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = conf.gpsts_config['Neural_Network']['EpochNumber']
    num_classes = conf.gpsts_config['Neural_Network']['ClassNumber']
    learning_rate = conf.gpsts_config['Neural_Network']['LearningRate']
    batch_size_train = conf.gpsts_config['Neural_Network']['BatchSizeTrain']
    batch_size_val = conf.gpsts_config['Neural_Network']['BatchSizeVal']
    batch_size_test = conf.gpsts_config['Neural_Network']['BatchSizeTest']
    expname = conf.gpsts_config['Experiment_Settings']['ExpName']
    
    data_train = DatasetFolder(conf.gpsts_config['Neural_Network']['TrainingPath']+'train/',
        loader = np_loader,
        extensions = ('.npy'))
    data_test = DatasetFolder(conf.gpsts_config['Neural_Network']['TrainingPath']+'validation/',
        loader = np_loader,
        extensions = ('.npy'))
    
    im_path = str(conf.im_path)+'/'+str(conf.gpsts_config["Experiment_Settings"]["ExpName"])+'/'
    
    #idx = list(range(len(data_test)))
    idx = np.load('/Users/jthomas/GitRepo/gpSTS_tut/data/gpSTS_spectroscopy/idx.npy',allow_pickle=True)
    test_split = int(0.4*(len(data_test)+1))
    #random.shuffle(idx)
    #np.save(im_path+'idx.npy',idx)
    test_idx, val_idx = idx[:test_split], idx[test_split:]
    data_val = dataloader.Subset(data_test, val_idx)
    data_test = dataloader.Subset(data_test, test_idx)

    model = Conv1d(num_classes).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = dataloader.DataLoader(data_train, batch_size=batch_size_train, shuffle=True)
    val_loader = dataloader.DataLoader(data_val, batch_size=batch_size_val, shuffle=True)
    losses = []
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    correct = 0
    total = 0
    print()
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        epochs.append(epoch+1)
        for i, (spectra, labels) in enumerate(train_loader):
            spectra = spectra.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(spectra)
            loss = criterion(outputs, labels)       
            # Store loss
            losses.append(loss.item())     
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()          
            predicted = outputs.max(1,keepdim=True)[1]
            total += labels.size(0)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()           
            # Update progress bar
            if (i+1) % 10 == 0:               
                progbar(i+1, total_step, 10, epoch+1, num_epochs, loss.item(), correct / total)
       
        #Eval Model
        model.eval() 
        
        #Train eval
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for data in train_loader:
                spec, labels = data
                outputs = model(spec.to(device))
                predicted = outputs.max(1,keepdim=True)[1]
                predicted = predicted.to(device)
                labels = labels.to(device)
                loss_loc = criterion(outputs.to(device), labels.to(device))
                loss += loss_loc.item()
                total += labels.size(0)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                losses.append(loss_loc.item())
                
            batch_accuracy = correct / total
            train_losses.append(float(np.mean(losses)))
            train_accuracies.append(100 * batch_accuracy)
            
        #Validation eval
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                spec, labels = data
                outputs = model(spec.to(device))
                predicted = outputs.max(1,keepdim=True)[1]
                predicted = predicted.to(device)
                labels = labels.to(device)
                loss_loc = criterion(outputs.to(device), labels.to(device))
                loss += loss_loc.item()
                total += labels.size(0)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                losses.append(loss_loc.item())
                
            batch_accuracy = correct / total
            val_losses.append(float(np.mean(losses)))
            val_accuracies.append(100 * batch_accuracy)

        # Create test dataloader to make predictions 
        test_loader = dataloader.DataLoader(data_test, batch_size=batch_size_test, shuffle=True)

        # Make predictions on unseen test data
        y_true, y_pred = make_predictions(model, device, test_loader)

        test_corr = 0
        for i in range(0,len(y_true)):
            if y_true[i] == y_pred[i]:
                test_corr += 1
        test_acc = test_corr/len(y_pred)

        print()
        print(f'Test Accuracy on the {str(len(y_pred))} test spectra: {100 * test_acc} %')
        print()
        torch.save(model.state_dict(), conf.gpsts_config['Neural_Network']['TrainingPath']+expname+'.ckpt')
        plot_metrics(im_path, y_true, y_pred, num_classes, epoch+1, epochs, {
        "loss": train_losses, 
        "accuracy": train_accuracies, 
        "val_loss": val_losses, 
        "val_accuracy": val_accuracies
        })
        #plot_cm(im_path,num_epochs, y_true, y_pred, num_classes=num_classes)
