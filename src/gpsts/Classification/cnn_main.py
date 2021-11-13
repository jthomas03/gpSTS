import torch 
import torch.nn as nn
import torch.utils.data as dataloader
import torchvision
from torchvision.datasets import DatasetFolder
from sklearn.model_selection import train_test_split
import random
import sys
import os
import numpy as np
import torch.nn.functional as F
import Config
import Config as conf
from gpsts.Classification.cnn1dimport import np_loader, Conv1d, progbar, make_predictions, dplot, spec_loader, dscplot


def main_cnn():
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = conf.nanonis_config['Neural_Network']['EpochNumber']
    num_classes = conf.nanonis_config['Neural_Network']['ClassNumber']
    learning_rate = conf.nanonis_config['Neural_Network']['LearningRate']
    batch_size_train = conf.nanonis_config['Neural_Network']['BatchSizeTrain']
    batch_size_val = conf.nanonis_config['Neural_Network']['BatchSizeVal']
    batch_size_test = conf.nanonis_config['Neural_Network']['BatchSizeTest']
    expname = conf.nanonis_config['Nanonis_Settings']['ExperimentName']
    model_name = expname.replace(' ',"_")
    
    data_train = DatasetFolder(conf.nanonis_config['Neural_Network']['TrainingPath']+'train//',
        loader = np_loader,
        extensions = ('.npy'))
    data_test = DatasetFolder(conf.nanonis_config['Neural_Network']['TrainingPath']+'validation//',
        loader = np_loader,
        extensions = ('.npy'))
    
    idx = list(range(len(data_test)))
    test_split = int(0.4*(len(data_test)+1))
    random.shuffle(idx)
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
    correct = 0
    total = 0
    print()
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
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

    # Test the model
    model.eval() 
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
        print()
        print(f'Test Accuracy on the {str(total)} validation spectra of batch size {str(batch_size_val)}: {100 * batch_accuracy} %')

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
    test = input("Save checkpoint file by entering a 1 or quit otherwise: ")
    if test != '1':
        print("gpSTS is exiting.")
        sys.exit(1)

    if len(sys.argv) == 1:
        torch.save(model.state_dict(), conf.nanonis_config['Neural_Network']['TrainingPath']+model_name+'.ckpt')

def main_class():
    dec = input("Plot interpolated results by entering a 1, plot collected point classification by entering a 2, or exit otherwise: ")
    deccorr = ['1','2']
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dpath = conf.nanonis_config['Nanonis_Settings']['DataLocation']
    expname = conf.nanonis_config['Nanonis_Settings']['ExperimentName']
    tpath = conf.nanonis_config['Neural_Network']['TrainingPath']
    num_classes = conf.nanonis_config['Neural_Network']['ClassNumber']
    model_name = expname.replace(' ',"_")
    assert os.path.isfile(dpath + model_name + '_3dinter.npy')
    assert os.path.isfile(dpath + model_name + '_3dsparse.npy')
    assert os.path.isfile(tpath + model_name + '.ckpt')
    out = np.load(dpath + model_name + '_3dsparse.npy', allow_pickle=True)
    out_full = np.load(dpath + model_name + '_3dinter.npy')
    outa_full = np.zeros((out_full.shape[0],out_full.shape[1]))
    if dec not in deccorr:
        print("gpSTS is exiting.")
        sys.exit(1)

    if dec == '1':
        print()
        model = Conv1d(num_classes).to(device)
        model.load_state_dict(torch.load(tpath + model_name+'.ckpt', map_location=device))
        model.eval()
        for i in range(0,out_full.shape[0]):
            for j in range(0,out_full.shape[1]):
                test_out = spec_loader(out_full[j][i])
                test_out = test_out.to(device)
                output = model(test_out)
                index = output.data.cpu().numpy().argmax()
                prob = F.softmax(output, dim=1)
                outputs = model(test_out)
                _, predicted = torch.max(outputs, 1)
                outa_full[j][i] = predicted[0].item()
        dplot(outa_full)

    if dec == '2':
        print()
        model = Conv1d(num_classes).to(device)
        model.load_state_dict(torch.load(tpath + model_name+'.ckpt', map_location=device))
        model.eval()
        scplotx = []
        scploty = []
        scplotdi = []
        for x, y, di in out:
            di_n = np.array(di)
            test_out = spec_loader(di_n)
            test_out = test_out.to(device)
            output = model(test_out)
            index = output.data.cpu().numpy().argmax()
            prob = F.softmax(output, dim=1)
            outputs = model(test_out)
            _, predicted = torch.max(outputs, 1)
            dipred = predicted[0].item()
            scplotx.append(x)
            scploty.append(y)
            scplotdi.append(dipred)
            
        dscplot(scplotx,scploty,scplotdi)

