import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import LPU.external_libs.PU_learning
import LPU.external_libs.PU_learning.algorithm
import LPU.external_libs.PU_learning.utils


def validate(epoch, net, u_validloader, criterion, device, threshold, logistic=True, show_bar=True, separate=False):
    
    if show_bar:     
        print('\nTest Epoch: %d' % epoch)
    
    net.eval() 
    test_loss = 0
    correct = 0
    total = 0

    pos_correct = 0
    neg_correct = 0

    pos_total = 0
    neg_total = 0

    if not logistic: 
        # print("here")
        criterion = LPU.external_libs.PU_learning.algorithm.sigmoid_loss

    with torch.no_grad():
        for batch_idx, (_, inputs, _, true_targets) in enumerate(u_validloader):
            
            inputs , true_targets = inputs.to(device), true_targets.to(device)
            outputs = net(inputs)
            

            predicted  = torch.nn.functional.softmax(outputs, dim=-1)[:,0] \
                    <= torch.tensor([threshold]).to(device)

            if not logistic: 
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                
            loss = criterion(outputs, true_targets)

            test_loss += loss.item()
            total += true_targets.size(0)
            
            correct_preds = predicted.eq(true_targets).cpu().numpy()
            correct += np.sum(correct_preds)

            if separate: 

                true_numpy = true_targets.cpu().numpy().squeeze()
                pos_idx = np.where(true_numpy==0)[0]
                neg_idx = np.where(true_numpy==1)[0]

                pos_correct += np.sum(correct_preds[pos_idx])
                neg_correct += np.sum(correct_preds[neg_idx])

                pos_total += len(pos_idx)
                neg_total += len(neg_idx)

            if show_bar: 
                LPU.external_libs.PU_learning.utils.progress_bar(batch_idx, len(u_validloader) , 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    if not separate: 
        return 100.*correct/total
    else: 
        return 100.*correct/total, 100.*pos_correct/pos_total, 100.*neg_correct/neg_total



def train(epoch, net, p_trainloader, u_trainloader, optimizer, criterion, device, show_bar=True):
    
    if show_bar:     
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    y_vals = []
    y_probs = []
    y_ests = []
    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        optimizer.zero_grad()
        _, p_inputs, p_targets = p_data
        _, u_inputs, u_targets, u_true_targets = u_data

        p_targets = p_targets.to(device)
        u_targets = u_targets.to(device)

       
        inputs =  torch.cat((p_inputs, u_inputs), dim=0)
        targets =  torch.cat((p_targets, u_targets), dim=0)
        inputs = inputs.to(device)

        outputs = net(inputs)

        p_outputs = outputs[:len(p_targets)]
        u_outputs = outputs[len(p_targets):]
        

        p_loss = criterion(p_outputs, p_targets)
        u_loss = criterion(u_outputs, u_targets)
        loss = (p_loss + u_loss)/2.0
        loss.backward()
        optimizer.step()

        outputs = torch.nn.functional.softmax(outputs, dim=-1)
        predicted = outputs[:,0] <= torch.tensor([0.5]).to(device)

        train_loss += loss.item()
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)
        y_vals.extend(u_true_targets.cpu().numpy())
        y_probs.extend(outputs[:,0].cpu().numpy())
        y_ests.extend(predicted.cpu().numpy())
        if show_bar: 
            LPU.external_libs.PU_learning.utils.progress_bar(batch_idx, len(p_trainloader) + len(u_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        

    return 100.*correct/total

def train_PU_discard(epoch, net,  p_trainloader, u_trainloader, optimizer, criterion, device, keep_sample=None, show_bar=True):
    
    if show_bar:     
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        
        optimizer.zero_grad()
        
        _, p_inputs, p_targets = p_data
        u_index, u_inputs, u_targets, u_true_targets = u_data

        u_idx = np.where(keep_sample[u_index.numpy()]==1)[0]

        if len(u_idx) <1: 
            continue

        u_targets = u_targets[u_idx]

        p_targets = p_targets.to(device)
        u_targets = u_targets.to(device)
        

        u_inputs = u_inputs[u_idx]        
        inputs =  torch.cat((p_inputs, u_inputs), dim=0)
        targets =  torch.cat((p_targets, u_targets), dim=0)
        inputs = inputs.to(device)

        outputs = net(inputs)

        p_outputs = outputs[:len(p_targets)]
        u_outputs = outputs[len(p_targets):]
        
        p_loss = criterion(p_outputs, p_targets)
        u_loss = criterion(u_outputs, u_targets)

        loss = (p_loss + u_loss)/2.0

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)

        if show_bar:
            LPU.external_libs.PU_learning.utils.progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100.*correct/total