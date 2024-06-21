
import copy

import numpy as np
import torch.nn
import LPU.external_libs.PU_learning.utils
import LPU.models.lpu_model_base
import LPU.constants
import LPU.utils
import LPU.utils.auxiliary_models

DEFAULT_CONFIG = {
    "dataset": "animal_no_animal",
    "input_dim": 4096,
    "set_seed": True,
    "lr": 0.01,
    "wd": 0.0005,
    "momentum": 0.9,
    "data_type": None,
    "train_method": "TEDn",
    "net_type": "LeNet",
    "sigmoid_loss": True,
    "estimate_alpha": True,
    "warm_start": True,
    "warm_start_epochs": 100,
    "epochs": 100,
    "alpha": 0.5,
    "beta": 0.5,
    "log_dir": "logging_accuracy",
    "data_dir": "data",
    "optimizer": "Adam",
    "alpha_estimate": 0.0,
    "show_bar": False,
    "use_alpha": False,
    "device": "cpu",
    "dataset_kind": "LPU",
    'ratios': 
    {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        'test': 0.4,
        'val': 0.05,
        'holdout': .05,
        'train': .5, 
    },
    "batch_size": {
        "train": 64,
        "test": 64,
        "val": 64,
        "holdout": 64
    },
    "data_generating_process": "SB"  # either of CC (case-control) or SB (selection-bias)
}


def create_dataloaders_dict_mpe(config, drop_last=False):
    # dataloders_dict = {}
    # samplers_dict = {}
    mpe_dataset_dict = {}
    mpe_dataloaders_dict = {}
    mpe_indices_dict = {}
    ratios_dict = config['ratios']
    data_generating_process = config['data_generating_process']
    data_type = LPU.constants.DTYPE
    if config['dataset_kind'] == 'LPU':
        lpu_dataset = LPU.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal')    
        l_y_cat_transformed = lpu_dataset.l.cpu().numpy() * 2 + lpu_dataset.y.cpu().numpy()
        split_indices_dict = LPU.utils.dataset_utils.index_group_split(np.arange(len(l_y_cat_transformed)), ratios_dict=ratios_dict, random_state=LPU.constants.RANDOM_STATE, strat_arr=l_y_cat_transformed)
        for split in split_indices_dict.keys():
            # *** DO NOT DELETE *** for the normal case where we have a LPU dataset
            # samplers_dict[split], dataloders_dict[split] = LPU.utils.dataset_utils.make_data_loader(lpu_dataset, batch_size=config['batch_size'][split],)
            mpe_dataset_dict[split], mpe_indices_dict[split] = LPU.utils.dataset_utils.LPUD_to_MPED(lpu_dataset=lpu_dataset, indices=split_indices_dict[split], data_generating_process=data_generating_process)
            mpe_dataloaders_dict[split] = {}
            for dataset_type in mpe_dataset_dict[split].keys():
                mpe_dataloaders_dict[split][dataset_type] = torch.utils.data.DataLoader(mpe_dataset_dict[split][dataset_type], batch_size=config['batch_size'][split], drop_last=drop_last, shuffle=True)
    elif config['dataset_kind'] == 'MPE':
        p_trainloader, u_trainloader, p_validloader, u_validloader, net, X, Y, p_validdata, u_validdata, u_traindata, p_traindata = \
                LPU.external_libs.PU_learning.helper.get_dataset(config['data_dir'], config['data_type'], config['net_type'], config['device'], config['alpha'], config['beta'], config['batch_size'])


        mpe_dataloaders_dict['train']= {}
        mpe_dataloaders_dict['test'] ={}
        mpe_dataloaders_dict['val'] = {}
        mpe_dataloaders_dict['holdout'] = {}

        mpe_dataloaders_dict['train']['PDataset'] = p_trainloader
        mpe_dataloaders_dict['train']['UDataset'] = u_trainloader

        mpe_dataloaders_dict['test']['PDataset'] = p_validloader
        mpe_dataloaders_dict['test']['UDataset'] = u_validloader

        mpe_dataloaders_dict['holdout']['PDataset'] = p_validloader
        mpe_dataloaders_dict['holdout']['UDataset'] = u_validloader


        mpe_dataloaders_dict['val']['PDataset'] = p_validloader
        mpe_dataloaders_dict['val']['UDataset'] = u_validloader
    else:
        raise ValueError("Dataset needs to be either LPU or MPE")
    return mpe_dataloaders_dict
    
class MPE(LPU.models.lpu_model_base.LPUModelBase):
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
        self.input_dim = config.get('input_dim', None)
        if self.input_dim is not None:
            self.initialize_model(self.input_dim)

    def initialize_model(self, dim):
        self.net = LPU.utils.auxiliary_models.MultiLayerPerceptron(input_dim=dim, output_dim=2).to(self.device).to(LPU.constants.DTYPE)
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
        scores_dict = {}
        all_ls = []
        all_ys = []
        all_inputs = []
        all_y_outputs = []
        all_l_outputs = []
        self.set_C(l_mean=.5)
        for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
            optimizer.zero_grad()
            _, p_inputs, p_targets = p_data
            _, u_inputs, u_targets, u_true_targets = u_data
            
            p_targets = p_targets.to(self.device)
            u_targets = u_targets.to(self.device)

            # breakpoint()
        
            inputs =  torch.cat((p_inputs, u_inputs), dim=0)
            targets =  torch.cat((p_targets, u_targets), dim=0)
            true_targets = torch.cat([p_targets, u_true_targets])
            inputs = inputs.to(self.device)


            outputs = self.net(inputs)

            if self.config['data_generating_process'] == 'SB':
                offset = 0
            else:
                offset = len(p_targets)
                
            y_batch = true_targets[offset:]
            l_batch = targets[offset:]
            inputs_batch = inputs[offset:]
            outputs_batch = outputs[offset:]


            if len(all_ls) == 0:
                all_ls = l_batch
                all_ys = y_batch
                all_inputs = inputs_batch
                all_y_outputs = outputs_batch
            else:
                all_ls = torch.concat((all_ls, l_batch), dim=0)
                all_ys = torch.concat((all_ys, y_batch), dim=0)
                all_inputs = torch.concat((all_inputs, inputs_batch), dim=0)
                all_y_outputs = torch.concat((all_y_outputs, outputs_batch), dim=0)
                
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
                LPU.external_libs.PU_learning.utils.progress_bar(batch_idx, len(p_trainloader) + len(u_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (total_loss/(batch_idx+1), 100.*correct/total_size, correct, total_size))
        all_y_outputs = torch.nn.functional.softmax(all_y_outputs, dim=-1)[:,1].detach().cpu().numpy().squeeze()
        # all_l_outputs = 1 - all_y_outputs
        all_l_outputs = all_y_outputs * self.C.detach().cpu().numpy()
        all_l_ests = all_l_outputs > 0.5 * self.C.detach().cpu().numpy()
        all_y_ests = all_y_outputs > 0.5
        all_scores = self._calculate_validation_metrics(y_probs=all_y_outputs, y_vals=all_ys, l_probs=all_l_outputs, l_vals=all_ls, l_ests=all_l_ests, y_ests=all_y_ests)
        all_scores['overall_loss'] = total_loss / (batch_idx + 1)
        return all_scores
            # else:
            #     scores_dict = {'loss': total_loss / (batch_idx + 1), 'accuracy': 100. * correct / total_size}
            # return scores_dict       
     # return 100.*correct/total_size
    
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
        
        # p_targets = torch.hstack([p_target for (_, _, p_target)  in copy.deepcopy(p_trainloader)])
        # u_p_targets = torch.hstack([u_target for (_, _, _, u_target)  in copy.deepcopy(u_trainloader)])
        # print ("positive ratio in training:", (torch.sum((p_targets==0)) + torch.sum(u_p_targets==0)) / (len(p_targets) + len(u_p_targets)))
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
        scores_dict = {}
        all_ls = []
        all_ys = []
        all_inputs = []
        all_y_outputs = []
        all_l_outputs = []
        self.set_C(l_mean=.5)

        for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):

            if update_gradient:
                optimizer.zero_grad()
            
            _, p_inputs, p_targets = p_data
            u_index, u_inputs, u_targets, u_true_targets = u_data

            u_idx = np.where(keep_samples[u_index.numpy()]==1)[0]

            if len(u_idx) <1: 
                continue

            inputs =  torch.cat((p_inputs, u_inputs), dim=0)
            inputs = inputs.to(self.device)
            outputs = self.net(inputs)

            y_concat = u_true_targets
            l_concat = torch.ones_like(u_targets)
            inputs_concat = u_inputs
            outputs_concat = outputs[len(p_targets):]
            

            if len(all_ls) == 0:
                all_ls = l_concat
                all_ys = y_concat
                all_inputs = inputs_concat
                all_y_outputs = outputs_concat
            else:
                all_ls = torch.concat((all_ls, l_concat), dim=0)
                all_ys = torch.concat((all_ys, y_concat), dim=0)
                all_inputs = torch.concat((all_inputs, inputs_concat), dim=0)
                all_y_outputs = torch.concat((all_y_outputs, outputs_concat), dim=0)


            p_targets = p_targets.to(self.device)
            u_targets = u_targets.to(self.device)
            targets =  torch.cat((p_targets, u_targets), dim=0)



            p_outputs = outputs[:len(p_targets)]
            u_outputs = outputs[len(p_targets):]
            
            u_targets_subset = u_targets[u_idx]
            u_outputs_subset = u_outputs[u_idx]

            p_loss = criterion(p_outputs, p_targets)
            u_loss = criterion(u_outputs_subset, u_targets_subset)

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
                LPU.external_libs.PU_learning.utils.progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total_size, correct, total_size))
                
        total_loss /= len(p_trainloader)                
        total_p_loss /= len(p_trainloader)
        total_u_loss /= len(u_trainloader)
        all_y_outputs = torch.nn.functional.softmax(all_y_outputs, dim=-1)[:,1].detach().cpu().numpy().squeeze()
        all_l_outputs = all_y_outputs * self.C.detach().cpu().numpy()
        all_l_ests = all_l_outputs > 0.5 * self.C.detach().cpu().numpy()
        all_y_ests = all_y_outputs > 0.5
        all_scores = self._calculate_validation_metrics(y_probs=all_y_outputs, y_vals=all_ys, l_probs=all_l_outputs, l_vals=all_ls, l_ests=all_l_ests, y_ests=all_y_ests)
        all_scores['overall_loss'] = total_loss
        all_scores['p_loss'] = total_p_loss
        all_scores['u_loss'] = total_u_loss
        return all_scores

        # return 100.*correct/total_size, loss_dict

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

        with torch.no_grad():
            for batch_idx,(_, X_u, _, y_u) in enumerate(u_validloader):
                # print ("positive ratio in validate:", (torch.sum((y_p==0)) + torch.sum(y_u==0)) / (len(y_p) + len(y_u)))
                X = X_u
                y = y_u

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
                    LPU.external_libs.PU_learning.utils.progress_bar(batch_idx, len(u_validloader) , 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total))

                # y = 1 - y
                # l = 1 - l
                # predicted = 1 - predicted.detach().cpu().numpy()
                outputs_probs = 1 - outputs_probs.detach().cpu().numpy()
                l_probs = self.predict_proba(X)
                l_probs = l_probs
                l_ests = self.predict(X, threshold=0.5 * self.C.detach().cpu().numpy())
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
        validation_results['overall_loss'] = total_loss/(batch_idx+1)
        return validation_results

    def predict_prob_y_given_X(self, X=None, f_x=None):
        if f_x is None:
            if type(X) == np.ndarray:
                X = torch.tensor(X, dtype=LPU.constants.DTYPE)
            X = X.to(self.device)
            f_x = self.net(X)
        predicted_prob  = torch.nn.functional.softmax(f_x, dim=-1)[:,1]
        return predicted_prob
    
        
    def set_C(self, l_mean):
        with torch.no_grad():
            self.C.fill_(l_mean * self.alpha_estimate)
    
    def predict_prob_l_given_y_X(self, X=None, f_x=None):
        return self.C.detach().cpu().numpy()
        

