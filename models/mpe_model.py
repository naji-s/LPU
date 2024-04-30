
import copy

import numpy as np
import torch.nn
import lpu.external_libs.PU_learning.utils
import lpu.models.lpu_model_base
import lpu.external_libs.PU_learning.estimator
import lpu.external_libs.PU_learning.train_PU
import lpu.constants

    
class MPE(lpu.models.lpu_model_base.LPUModelBase):
    def __init__(self, config):
        super(MPE, self).__init__()
        self.config = config
        self.device = config['device']
        self.true_alpha = config['alpha']
        self.beta = config['beta']
        self.show_bar = config['show_bar']
        self.use_alpha = config['use_alpha']
        if self.use_alpha:
            self.alpha_estimate = self.true_alpha
        else:
            self.alpha_estimate = config['alpha_estimate']
    def initialize_model(self, dim):
        self.net = MultiLayerPerceptron(dim=dim).to(self.device).to(lpu.constants.DTYPE)
        if self.config['device'].startswith('cuda'):
            self.net = torch.nn.DataParallel(self.net)
            torch.cudnn.benchmark = True

    def p_probs(self, p_loader): 
        self.net.eval()
        pp_probs = None
        with torch.no_grad():
            for batch_idx, (_, inputs, targets) in enumerate(p_loader):
            
                inputs = inputs.to(self.device)
                outputs = self.net(inputs)

                probs = torch.nn.functional.softmax(outputs, dim=-1)[:,0] 
                if pp_probs is None: 
                    pp_probs = probs.detach().cpu().numpy().squeeze()
                else:
                    pp_probs = np.concatenate((pp_probs, \
                        probs.detach().cpu().numpy().squeeze()), axis=0)
        
        return pp_probs    

    def u_probs(self, u_loader):
        self.net.eval()
        pu_probs = None
        pu_targets = None
        with torch.no_grad():
            for batch_idx, (_, inputs, _, targets) in enumerate(u_loader):
                inputs = inputs.to(self.device)
                outputs = self.net(inputs)
                    
                probs = torch.nn.functional.softmax(outputs, dim=-1) 
                
                if pu_probs is None: 
                    pu_probs = probs.detach().cpu().numpy().squeeze()
                    pu_targets = targets.numpy().squeeze()
                    
                else:
                    pu_probs = np.concatenate( (pu_probs, \
                        probs.detach().cpu().numpy().squeeze()))
                    pu_targets = np.concatenate( (pu_targets, \
                        targets.numpy().squeeze()))
                    
        
        return pu_probs, pu_targets

    def DKW_bound(self, x,y,t,m,n,delta=0.1, gamma= 0.01):

        temp = np.sqrt(np.log(4/delta)/2/n) + np.sqrt(np.log(4/delta)/2/m)
        bound = temp*(1+gamma)/(y/n)

        estimate = t

        return estimate, t - bound, t + bound


    def BBE_estimator(self, pdata_probs, udata_probs, udata_targets):

        p_indices = np.argsort(pdata_probs)
        sorted_p_probs = pdata_probs[p_indices]
        u_indices = np.argsort(udata_probs[:,0])
        sorted_u_probs = udata_probs[:,0][u_indices]
        sorted_u_targets = udata_targets[u_indices]

        sorted_u_probs = sorted_u_probs[::-1]
        sorted_p_probs = sorted_p_probs[::-1]
        sorted_u_targets = sorted_u_targets[::-1]
        num = len(sorted_u_probs)

        estimate_arr = []

        upper_cfb = []
        lower_cfb = []            

        i = 0
        j = 0
        num_u_samples = 0

        while (i < num):
            start_interval = sorted_u_probs[i]   
            k = i 
            if (i<num-1 and start_interval > sorted_u_probs[i+1]): 
                pass
            else: 
                i += 1
                continue
            if (sorted_u_targets[i]==1):
                num_u_samples += 1

            while ( j<len(sorted_p_probs) and sorted_p_probs[j] >= start_interval):
                j+= 1

            if j>1 and i > 1:
                t = (i)*1.0*len(sorted_p_probs)/j/len(sorted_u_probs)
                estimate, lower , upper = self.DKW_bound(i, j, t, len(sorted_u_probs), len(sorted_p_probs))
                estimate_arr.append(estimate)
                upper_cfb.append( upper)
                lower_cfb.append( lower)
            i+=1

        if (len(upper_cfb) != 0): 
            idx = np.argmin(upper_cfb)
            mpe_estimate = estimate_arr[idx]

            return mpe_estimate, lower_cfb, upper_cfb
        else: 
            return 0.0, 0.0, 0.0

        
    def estimate_alpha(self, p_holdoutloader, u_holdoutloader):
        """
        Estimates the alpha value using the BBE estimator.

        NOTE: in the original code (https://github.com/acmi-lab/PU_learning/blob/5e5e350dc0588de95a36eb952e3cf5382e786aec/train_PU.py#L130)
        alpha is estimated using validation set, which is also used as the test set. This is not a good practice, hence below we use the holdout set.

        Args:
            net (object): The neural network model.
            self.config (dict): Configuration parameters.
            p_holdoutloader (object): DataLoader for positive holdout data.
            u_holdoutloader (object): DataLoader for unlabeled holdout data.

        Returns:
            float: The estimated alpha value.

        Raises:
            None

        """
        p_probs = self.p_probs(p_holdoutloader)
        unlabeled_probs, unlabeled_targets = self.u_probs(u_holdoutloader)
        mpe_estimate, _, _ = self.BBE_estimator(p_probs, unlabeled_probs, unlabeled_targets)
        return mpe_estimate
    

    def warm_up_one_epoch(self, epoch, p_trainloader, u_trainloader, optimizer, criterion, valid_loader):
        if self.show_bar:     
            print('\nTrain Epoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total_size = 0
        total_loss = 0

        for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
            optimizer.zero_grad()
            _, p_inputs, p_targets = p_data
            _, u_inputs, u_targets, u_true_targets = u_data


            
            p_targets = p_targets.to(self.device)
            u_targets = u_targets.to(self.device)

        
            inputs =  torch.cat((p_inputs, u_inputs), dim=0)
            targets =  torch.cat((p_targets, u_targets), dim=0)
            inputs = inputs.to(self.device)

            outputs = self.net(inputs)

            p_outputs = outputs[:len(p_targets)]
            u_outputs = outputs[len(p_targets):]
            

            p_loss = criterion(p_outputs, p_targets)
            u_loss = criterion(u_outputs, u_targets)
            loss = (p_loss + u_loss)/2.0
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_size += targets.size(0)
            
            correct_preds = predicted.eq(targets).cpu().numpy()
            correct += np.sum(correct_preds)

            if self.show_bar: 
                lpu.external_libs.PU_learning.utils.progress_bar(batch_idx, len(p_trainloader) + len(u_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (total_loss/(batch_idx+1), 100.*correct/total_size, correct, total_size))
        return 100.*correct/total_size
    
    def rank_inputs(self, epoch, u_trainloader, u_size):

        self.net.eval() 
        output_probs = np.zeros(u_size)
        keep_samples = np.ones_like(output_probs)
        true_targets_all = np.zeros(u_size)

        with torch.no_grad():
            for batch_num, (idx, inputs, _, true_targets) in enumerate(u_trainloader):
                idx = idx.numpy()
                inputs = inputs.to(self.device)
                outputs = self.net(inputs)


                probs  = torch.nn.functional.softmax(outputs, dim=-1)[:,0]         
                output_probs[idx] = probs.detach().cpu().numpy().squeeze()
                true_targets_all[idx] = true_targets.numpy().squeeze()

        sorted_idx = np.argsort(output_probs)

        keep_samples[sorted_idx[u_size - int(self.alpha_estimate*u_size):]] = 0

        neg_reject = np.sum(true_targets_all[sorted_idx[u_size - int(self.alpha_estimate*u_size):]]==1.0)

        neg_reject = neg_reject/ int(self.alpha_estimate*u_size)
        return keep_samples, neg_reject

    def train_one_epoch(self, epoch, p_trainloader, u_trainloader, optimizer, criterion, train_unlabeled_size, update_gradient=True): 
        keep_samples, _ = self.rank_inputs(epoch, u_trainloader, u_size=train_unlabeled_size)
        
        p_targets = torch.hstack([p_target for (_, _, p_target)  in copy.deepcopy(p_trainloader)])
        u_p_targets = torch.hstack([u_target for (_, _, _, u_target)  in copy.deepcopy(u_trainloader)])
        print ("positive ratio in training:", (torch.sum((p_targets==0)) + torch.sum(u_p_targets==0)) / (len(p_targets) + len(u_p_targets)))
        # breakpoint()
        if self.show_bar:
            print('\nTrain Epoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total_loss = 0
        total_p_loss = 0
        total_u_loss = 0
        total_size = 0
        for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
            if update_gradient:
                optimizer.zero_grad()
            
            _, p_inputs, p_targets = p_data
            u_index, u_inputs, u_targets, u_true_targets = u_data

            u_idx = np.where(keep_samples[u_index.numpy()]==1)[0]

            if len(u_idx) <1: 
                continue

            u_targets = u_targets[u_idx]

            p_targets = p_targets.to(self.device)
            u_targets = u_targets.to(self.device)
            

            u_inputs = u_inputs[u_idx]        
            inputs =  torch.cat((p_inputs, u_inputs), dim=0)
            targets =  torch.cat((p_targets, u_targets), dim=0)
            inputs = inputs.to(self.device)

            outputs = self.net(inputs)

            p_outputs = outputs[:len(p_targets)]
            u_outputs = outputs[len(p_targets):]
            
            p_loss = criterion(p_outputs, p_targets)
            u_loss = criterion(u_outputs, u_targets)

            loss = (p_loss + u_loss)/2.0
            total_p_loss += p_loss.item() / 2.
            total_u_loss += u_loss.item() / 2.
            total_loss += loss.item()
            if update_gradient:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            total_size += targets.size(0)
            
            correct_preds = predicted.eq(targets).cpu().numpy()
            correct += np.sum(correct_preds)

            if self.show_bar:
                lpu.external_libs.PU_learning.utils.progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total_size, correct, total_size))
        total_loss /= len(p_trainloader)                
        total_p_loss /= len(p_trainloader)
        total_u_loss /= len(u_trainloader)
        loss_dict = {'total_loss': total_loss, 'p_loss': total_p_loss, 'u_loss': total_u_loss}
        return 100.*correct/total_size, loss_dict

    def validate(self, epoch, p_validloader, u_validloader, criterion, threshold, separate=False, logistic=True):
        
        if self.show_bar:     
            print('\nTest Epoch: %d' % epoch)
        
        self.net.eval() 
        total_loss = 0
        correct = 0
        total = 0

        pos_correct = 0
        neg_correct = 0

        pos_total = 0
        neg_total = 0
        y_probs = []
        l_probs = []
        y_vals = []
        l_vals = []
        y_ests = []
        l_ests = []
        losses = []

        # if not logistic: 
            # print("here")
            # criterion = sigmoid_loss
        # print ("The balance for validation:", np.mean([true_targets==0 for _, inputs, _, true_targets in copy.deepcopy(u_validloader)]))
        with torch.no_grad():
            for batch_idx, ((_, X_p, y_p), (_, X_u, _, y_u)) in enumerate(zip(p_validloader, u_validloader)):
                # print ("positive ratio in validate:", (torch.sum((y_p==0)) + torch.sum(y_u==0)) / (len(y_p) + len(y_u)))
                # X = X_u
                # y = y_u
                X = torch.concat((X_p, X_u), dim=0)
                y = torch.concat((y_p, y_u), dim=0)
                X , y = X.to(self.device), y.to(self.device)
                outputs = self.net(X)
                outputs_probs = torch.nn.functional.softmax(outputs, dim=-1)[:,0]
                

                predicted  = outputs_probs <= torch.tensor([threshold]).to(self.device)

                if not logistic: 
                    outputs = torch.nn.functional.softmax(outputs, dim=-1)
                    
                loss = criterion(outputs, y)

                total_loss += loss.item()
                total += y.size(0)
                
                correct_preds = predicted.eq(y).cpu().numpy()
                correct += np.sum(correct_preds)

                if separate: 

                    true_numpy = y.cpu().numpy().squeeze()
                    pos_idx = np.where(true_numpy==0)[0]
                    neg_idx = np.where(true_numpy==1)[0]

                    pos_correct += np.sum(correct_preds[pos_idx])
                    neg_correct += np.sum(correct_preds[neg_idx])

                    pos_total += len(pos_idx)
                    neg_total += len(neg_idx)

                if self.show_bar: 
                    lpu.external_libs.PU_learning.utils.progress_bar(batch_idx, len(u_validloader) , 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total))

                # y = 1 - y
                # l = 1 - l
                # predicted = 1 - predicted.detach().cpu().numpy()
                outputs_probs = 1 - outputs_probs.detach().cpu().numpy()
                l_probs = self.predict_proba(X)
                l_probs = l_probs
                l_ests = self.predict(X)
                l_vals = np.zeros_like(y)
                y_vals.append(y)
                y_ests.append(predicted)
                y_probs.append(outputs_probs)
                losses.append(total_loss)
            # breakpoint()

        y_probs = np.hstack(y_probs)
        y_vals = np.hstack(y_vals).astype(int)
        l_probs = np.hstack(l_probs)
        l_vals = np.hstack(l_vals)
        l_ests = np.hstack(l_ests).astype(int)
        y_ests = np.hstack(y_ests).astype(int)

        validation_results = self._calculate_validation_metrics(
            y_probs, y_vals, l_probs, l_vals, l_ests=l_ests, y_ests=y_ests)
        # validation_results.update({'overall_loss': np.mean(losses)})
        # losses = losses[-1]
        validation_results['L_mpe'] = np.mean(losses)
        return validation_results


    def predict_prob_y_given_X(self, X):
        self.net.eval()
        with torch.no_grad():
            if type(X) == np.ndarray:
                X = torch.tensor(X, dtype=lpu.constants.DTYPE)
            X = X.to(self.device)
            outputs = self.net(X)
            predicted_prob  = torch.nn.functional.softmax(outputs, dim=-1)[:,1]
        return predicted_prob.cpu().numpy()
    
        
    def set_C(self, l_mean):
        self.C = l_mean * self.alpha_estimate
    
    def predict_prob_l_given_y_X(self, X):
        return self.C
        
