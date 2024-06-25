"""This is an implementation of https://github.com/VITA-Group/Self-PU/blob/a0e332ae4f8110e2490d597876e36bf837e1060f/train_2s2t_mix.py
in OOP form. 
"""
import os
import sys
import time
import tqdm



import sklearn.metrics
import torch.utils.data
import unittest.mock




import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import LPU.constants
import LPU.external_libs
import LPU.models.geometric.Elkan.Elkan
import LPU.models.lpu_model_base
import LPU.external_libs.Self_PU.datasets
import LPU.external_libs.Self_PU.functions
import LPU.external_libs.Self_PU.mean_teacher
import LPU.external_libs.Self_PU.meta_models
import LPU.external_libs.Self_PU.models
import LPU.external_libs.Self_PU.utils
import LPU.utils.utils_general

DEFAULT_CONFIG = {
    # "dataset_name": "animal_no_animal",
    'ratios': 
    {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        'test': 0.4,
        'val': 0.05,
        'holdout': .0,
        'train': .55, 
    },

    "batch_size": {
        "train": 64,
        "test": 64,
        "val": 64,
        "holdout": 64
    },
    # "batch_size": {
    #     "train": 8192,
    #     "test": 8192,
    #     "val": 8192,
    #     "holdout": 8192
    # },
    "dataset_kind": "LPU",
    "dataset_name": "animal_no_animal",
    "dim": 4096,
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
    "device": "cpu",
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.005,
    "modeldir": "LPU/scripts/selfPU/checkpoints/",
    "epochs": 10,
    "loss": "nnPU",
    "gpu": None,
    "workers": 0,
    "weight": 1.0,
    "self_paced": True,
    "self_paced_start": 10,
    "self_paced_stop": 50,
    "self_paced_frequency": 10,
    "self_paced_type": "A",
    "increasing": True,
    "replacement": True,
    "mean_teacher": True,
    "ema_start": 50,
    "ema_decay": 0.999,
    "consistency": 0.3,
    "consistency_rampup": 400,
    "top1": 0.4,
    "top2": 0.6,
    "soft_label": False,
    "datapath": "./data",
    "type": "mu",
    "alpha": 0.1,
    "gamma": 0.0625,
    "num_p": 1000

}

torch.set_default_dtype(LPU.constants.DTYPE)

LOG = LPU.utils.utils_general.configure_logger(__name__)

EPSILON = 1e-16


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        #print(val, n)
        if self.count == 0:
            self.avg = 0
        else:
            self.avg = self.sum / self.count

def accuracy(output, target):
    with torch.no_grad():
        
        batch_size = float(target.size(0))
        
        output = output.view(-1)
        correct = torch.sum(output == target).float()
        
        pcorrect = torch.sum(output[target==1] == target[target == 1]).float()
        ncorrect = correct - pcorrect
    
    ptotal = torch.sum(target == 1).float()

    if ptotal == 0:
        return torch.tensor(0.).to(target.device), ncorrect / (batch_size - ptotal) * 100, correct / batch_size * 100, ptotal
    elif ptotal == batch_size:
        return pcorrect / ptotal * 100, torch.tensor(0.).to(target.device), correct / batch_size * 100, ptotal
    else:
        return pcorrect / ptotal * 100, ncorrect / (batch_size - ptotal) * 100, correct / batch_size * 100, ptotal

class selfPU(LPU.models.lpu_model_base.LPUModelBase):
    """
    """
    def __init__(self, config, switched=False, *args, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # self.hidden_dim = self.config.get('hidden_dim', 32)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.model = None
        self.epochs = self.config.get('epochs', 200)
        self.ema_decay = None
        self.soft_label = self.config.get('soft_label', False)
        self.num_workers = self.config.get('num_workers', 0)
        self.model_dir = self.config.get('model_dir', 'LPU/scripts/selfPU')
        self.dim = self.config.get('dim', 784)
        self.switched = switched    
        self.mean_teacher_step = 0
        self.model1 = None
        self.model2 = None
        self.ema_model1 = None
        self.ema_model2 = None

    def create_meta_mlp(self):
        return LPU.external_libs.Self_PU.meta_models.MetaMLP(dim=self.dim)
    
    def create_model(self, ema=False):
        model_dict = {'mnist': self.create_meta_mlp, 'cifar': LPU.external_libs.Self_PU.meta_models.MetaCNN, 'animal_no_animal':self.create_meta_mlp}
        model = model_dict[self.config['dataset_name']]()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.to(self.config['device'])
        return torch.autograd.Variable(x, requires_grad=requires_grad)
    
    def check_noisy(self, epoch):
        if epoch >= self.config['self_paced_start']: #and args.turnoff_noisy:
            return False
        else:
            return True
        
    def set_C(self, holdout_dataloader):
        X_all = []
        l_all = []
        y_all = []
        for i, (X, l, y, _) in enumerate(holdout_dataloader):
            X_all.append(X)
            l_all.append(l)
            y_all.append(y)
        X = torch.cat(X_all, dim=0)
        l = torch.cat(l_all, dim=0)
        y = torch.cat(y_all, dim=0)
        self.C = l[y == 1].mean().detach().cpu().numpy()

    def check_mean_teacher(self, epoch):
        if not self.config['mean_teacher']:
            return False
        elif epoch < self.config['ema_start']:
            return False 
        else:
            return True
        
    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.config['consistency'] * sigmoid_rampup(epoch, self.config['consistency_rampup'])

    def update_ema_variables(self, model, ema_model):
        alpha = self.config['ema_decay']
        alpha = min(1 - 1 / (self.mean_teacher_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def check_self_paced(self, epoch):
        if self.config['self_paced_stop'] < 0: self_paced_stop = self.config['num_epochs']
        else: self_paced_stop = self.config['self_paced_stop']
        if not self.config['self_paced']:
            return False
        elif self.config['self_paced'] and epoch >= self_paced_stop:
            return False
        elif self.config['self_paced'] and epoch < self.config['self_paced_start']:
            return False
        else: return True

    # def predict_proba(self, X):
    #     self.model.eval()
    #     return torch.sigmoid(self.model(torch.as_tensor(X, dtype=LPU.constants.DTYPE))).detach().numpy().flatten()

    def predict_prob_y_given_X(self, X):
        return self.predict_proba(X) / self.C
    
    def predict_prob_l_given_y_X(self, X):
        return self.C 

    def loss_fn(self, X_batch, l_batch):
        return torch.tensor(0.0)
    
    # def update_dataset(self, student, teacher, dataset_train_clean, dataset_train_noisy, epoch, ratio=0.5, lt = 0, ht = 1):

    #     dataset_train_noisy.reset_ids()
    #     dataset_train_noisy.set_type("clean")
    #     dataloader_train = torch.utils.data.DataLoader(dataset_train_noisy, batch_size=self.config['batch_size']['train'], num_workers=4, shuffle=False, pin_memory=True)
    #     if self.config['dataset_name'] == 'mnist':
    #         results = np.zeros(dataloader_train.dataset.ids.max() + 1) # rid.imageid: p_pos # 存储概率结果
    #     elif self.config['dataset_name'] == 'cifar':
    #         results = np.zeros(51000) 
    #     else:
    #         results = np.zeros(len(dataset_train_noisy))
    #     student.eval()
    #     teacher.eval()
    #     # validation #######################
    #     with torch.no_grad():
    #         for i_batch, (X, _, _, T, ids, _) in enumerate(tqdm.tqdm(dataloader_train)):
    #             #images, lefts, rights, ages, genders, edus, apoes, labels, pu_labels, ids = Variable(sample_batched['mri']).to(self.config['device']), Variable(sample_batched['left']).to(self.config['device']), Variable(sample_batched['right']).to(self.config['device']), Variable(sample_batched['age']).to(self.config['device']), Variable(sample_batched['gender']).to(self.config['device']), Variable(sample_batched['edu']).to(self.config['device']), Variable(sample_batched['apoe']).to(self.config['device']), Variable(sample_batched['label']).view(-1).type(torch.LongTensor).to(self.config['device']), Variable(sample_batched['pu_label']).view(-1).type(torch.LongTensor).to(self.config['device']), sample_batched['id']
    #             X = X.to(self.config['gpu'])
    #             if self.config['dataset_name'] == 'mnist':
    #                 X = X.reshape(X.shape[0], 1, -1)
    #             # ===================forward====================
    #             outputs_s = student(X)
    #             outputs_t = teacher(X)
    #             prob_n_s = torch.sigmoid(outputs_s).view(-1).cpu().numpy()       
    #             prob_n_t = torch.sigmoid(outputs_t).view(-1).cpu().numpy()
    #             #print(np.sum(prob_n_t < 0.5))

    #             if self.check_mean_teacher(epoch):
    #                 results[ids.view(-1).numpy()] = prob_n_t
    #             else:
    #                 try:
    #                     results[ids.view(-1).numpy()] = prob_n_s
    #                 except:
    #                     breakpoint()
    #     # adni_dataset_train.update_labels(results, ratio)
    #     # dataset_origin = dataset_train
    #     ids_noisy = dataset_train_clean.update_ids(results, epoch, ratio = ratio, ht = ht, lt = lt) # 返回的是noisy ids
    #     dataset_train_noisy.set_ids(ids_noisy) # 将noisy ids更新进去
    #     dataset_train_noisy.set_type("noisy")
    #     dataset_train_noisy.update_prob(results)
    #     dataset_train_clean.update_prob(results)
    #     assert np.all(dataset_train_noisy.ids == ids_noisy) # 确定更新了
    #     #dataloader_origin = DataLoader(dataset_origin, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    #     dataloader_train_clean = torch.utils.data.DataLoader(dataset_train_clean, batch_size=self.config['batch_size']['train'], num_workers=self.config['num_workers'], shuffle=True, pin_memory=True)
    #     dataloader_train_noisy = torch.utils.data.DataLoader(dataset_train_noisy, batch_size=self.config['batch_size']['train'], num_workers=self.config['num_workers'], shuffle=False, pin_memory=True)

    #     return dataloader_train_clean, dataloader_train_noisy

    def train_with_meta(self, clean1_loader, noisy1_loader, clean2_loader, noisy2_loader, test_loader, model1, model2, ema_model1, ema_model2, criterion, consistency_criterion, optimizer1, scheduler1, optimizer2, scheduler2, epoch, warmup = False, self_paced_pick = 0):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        #losses = AverageMeter()
        pacc1 = AverageMeter()
        nacc1 = AverageMeter()
        pnacc1 = AverageMeter()
        pacc2 = AverageMeter()
        nacc2 = AverageMeter()
        pnacc2 = AverageMeter()
        pacc3 = AverageMeter()
        nacc3 = AverageMeter()
        pnacc3 = AverageMeter()
        pacc4 = AverageMeter()
        nacc4 = AverageMeter()
        pnacc4 = AverageMeter()
        count_clean = AverageMeter()
        count_noisy = AverageMeter()
        model1.train()
        model2.train()
        ema_model1.train()
        ema_model2.train()
        end = time.time()
        w1 = AverageMeter()
        w2 = AverageMeter()

        entropy_clean = AverageMeter()
        entropy_noisy = AverageMeter()

        count2 = AverageMeter()
        count1 = AverageMeter()

        binary_kind = set(np.unique(noisy1_loader.dataset.Y))

        consistency_weight = self.get_current_consistency_weight(epoch - 30)
        if not warmup: 
            scheduler1.step()
            scheduler2.step()
        resultt = np.zeros(61000)

        dataloader_test = iter(test_loader)

        if clean1_loader: 
            for i, (X, _, Y, T, ids, _) in enumerate(clean1_loader):
            # measure data loading time

                data_time.update(time.time() - end)
                X = X.to(self.config['device'])
                if self.config['dataset_name'] == 'mnist':
                    X = X.view(X.shape[0], -1)
                Y = Y.to(self.config['device']).float()
                T = T.to(self.config['device']).long()
                # compute output
                output1 = model1(X)
                output2 = model2(X)
                with torch.no_grad():
                    ema_output1 = ema_model1(X)

                consistency_loss = consistency_weight * \
                consistency_criterion(output1, ema_output1) / X.shape[0]

                predictiont1   = torch.sign(ema_output1).long()
                predictions1 = torch.sign(output1).long() # 否则使用自己的结果
                smx1 = torch.sigmoid(output1) # 计算sigmoid概率
                smx1 = torch.cat([1 - smx1, smx1], dim=1) # 组合成预测变量
                smxY = ((Y + 1) // 2).long() # 分类结果，0-1分类
                smx2 = torch.sigmoid(output2) # 计算sigmoid概率
                smx2 = torch.cat([1 - smx2, smx2], dim=1) # 组合成预测变量

                if self.config['soft_label']:
                    aux1 = - torch.mean(torch.sum(smx1 * torch.log(smx1 + 1e-10), dim = 1))
                    aux2 = - torch.mean(torch.sum(smx2 * torch.log(smx2 + 1e-10), dim = 1))
                else:
                    smxY = smxY.float()
                    smxY = smxY.view(-1, 1)
                    smxY = torch.cat([1 - smxY, smxY], dim = 1)
                    aux1 = - torch.mean(torch.sum(smxY * torch.log(smx1 + 1e-10), dim = 1)) # 计算Xent loss
                    aux2 = - torch.mean(torch.sum(smxY * torch.log(smx2 + 1e-10), dim = 1)) # 计算Xent loss

                
                loss = aux1
                entropy_clean.update(aux1, 1)

                if self.check_mean_teacher(epoch):
                    loss += consistency_loss

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                
                pacc_1, nacc_1, pnacc_1, psize = accuracy(predictions1, T) # 使用T来计算预测准确率
                pacc_3, nacc_3, pnacc_3, psize = accuracy(predictiont1, T) 
                pacc1.update(pacc_1, psize)
                nacc1.update(nacc_1, X.size(0) - psize)
                pnacc1.update(pnacc_1, X.size(0))
                pacc3.update(pacc_3, psize)
                nacc3.update(nacc_3, X.size(0) - psize)
                pnacc3.update(pnacc_3, X.size(0))
            
        if clean2_loader: 
            for i, (X, _, Y, T, ids, _) in enumerate(clean2_loader):
            # measure data loading time
            
                data_time.update(time.time() - end)
                X = X.to(self.config['device'])
                if self.config['dataset_name'] == 'mnist':
                    X = X.view(X.shape[0], -1)
                Y = Y.to(self.config['device']).float()
                T = T.to(self.config['device']).long()
                # compute output
                output1 = model1(X)
                output2 = model2(X)
                with torch.no_grad():
                    ema_output2 = ema_model2(X)

                consistency_loss = consistency_weight * \
                consistency_criterion(output2, ema_output2) / X.shape[0]

                predictiont2 = torch.sign(ema_output2).long()
                predictions2 = torch.sign(output2).long()

                smx1 = torch.sigmoid(output1) # 计算sigmoid概率
                smx1 = torch.cat([1 - smx1, smx1], dim=1) # 组合成预测变量

                smxY = ((Y + 1) // 2).long() # 分类结果，0-1分类

                smx2 = torch.sigmoid(output2) # 计算sigmoid概率
                smx2 = torch.cat([1 - smx2, smx2], dim=1) # 组合成预测变量

                if self.config['soft_label']:
                    aux1 = - torch.mean(torch.sum(smx1 * torch.log(smx1 + 1e-10), dim = 1))
                    aux2 = - torch.mean(torch.sum(smx2 * torch.log(smx2 + 1e-10), dim = 1))
                else:
                    smxY = smxY.float()
                    smxY = smxY.view(-1, 1)
                    smxY = torch.cat([1 - smxY, smxY], dim = 1)
                    aux1 = - torch.mean(torch.sum(smxY * torch.log(smx1 + 1e-10), dim = 1)) # 计算Xent loss
                    aux2 = - torch.mean(torch.sum(smxY * torch.log(smx2 + 1e-10), dim = 1)) # 计算Xent loss

                loss = aux2
                entropy_clean.update(aux2, 1)

                if self.check_mean_teacher(epoch):
                    loss += consistency_loss
                    
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()

                pacc_2, nacc_2, pnacc_2, psize = accuracy(predictions2, T)
                pacc_4, nacc_4, pnacc_4, psize = accuracy(predictiont2, T) 
                pacc2.update(pacc_2, psize)
                nacc2.update(nacc_2, X.size(0) - psize)
                pnacc2.update(pnacc_2, X.size(0))
                pacc4.update(pacc_4, psize)
                nacc4.update(nacc_4, X.size(0) - psize)
                pnacc4.update(pnacc_4, X.size(0))
                
        if self.check_mean_teacher(epoch):
            self.update_ema_variables(model1, ema_model1) # 更新ema参数
            self.update_ema_variables(model2, ema_model2)
            self.mean_teacher_step += 1
        LOG.info('Epoch Clean : [{0}]\t'
                'PACC1 {pacc1.val:.3f} ({pacc1.avg:.3f})\t'
                'NACC1 {nacc1.val:.3f} ({nacc1.avg:.3f})\t'
                'PNACC1 {pnacc1.val:.3f} ({pnacc1.avg:.3f})\t'
                'PACC2 {pacc2.val:.3f} ({pacc2.avg:.3f})\t'
                'NACC2 {nacc2.val:.3f} ({nacc2.avg:.3f})\t'
                'PNACC2 {pnacc2.val:.3f} ({pnacc2.avg:.3f})\t'
                'PACC3 {pacc3.val:.3f} ({pacc3.avg:.3f})\t'
                'NACC3 {nacc3.val:.3f} ({nacc3.avg:.3f})\t'
                'PNACC3 {pnacc3.val:.3f} ({pnacc3.avg:.3f})\t'
                'PACC4 {pacc4.val:.3f} ({pacc4.avg:.3f})\t'
                'NACC4 {nacc4.val:.3f} ({nacc4.avg:.3f})\t'
                'PNACC4 {pnacc4.val:.3f} ({pnacc4.avg:.3f})\t'.format(
                epoch, pacc1=pacc1, nacc1=nacc1, pnacc1=pnacc1, 
                pacc2=pacc2, nacc2=nacc2, pnacc2=pnacc2, pacc3=pacc3, nacc3=nacc3, pnacc3=pnacc3, 
                pacc4=pacc4, nacc4=nacc4, pnacc4=pnacc4))
        #if epoch > self.config['self_paced_start']: criterion.update_p(0.05)
        if (self.config['dataset_name'] == 'cifar'):
            criterion.update_p((20000 - self_paced_pick / 2) / (50000 - self_paced_pick))
            LOG.info("Setting Pi_P to {}".format((20000 - self_paced_pick / 2) / (50000 - self_paced_pick)))

        l_batch_concat = []
        y_batch_concat = []
        y_batch_concat_prob = []
        l_batch_concat_prob = []
        l_batch_concat_est = []
        y_batch_concat_est = []
        loss_batch_concat = []
        for i, (X, _, Y, T, ids, _) in enumerate(noisy1_loader):
            meta_net = self.create_model()
            meta_net.load_state_dict(model1.state_dict())

            if torch.cuda.is_available():
                meta_net.to(self.config['device'])
            #print(torch.max(X))
            X = X.to(self.config['device'])
            if self.config['dataset_name'] == 'mnist':
                X = X.view(X.shape[0], -1)
            Y = Y.to(self.config['device']).float()
            T = T.to(self.config['device']).long()
            
            y_f_hat = meta_net(X)
            prob = torch.sigmoid(y_f_hat)
            prob = torch.cat([1-prob, prob], dim=1)

            cost1 = torch.sum(prob * torch.log(prob + 1e-10), dim = 1)
            eps = self.to_var(torch.zeros(cost1.shape[0], 2))
            cost2 = criterion(y_f_hat, Y, eps = eps[:, 0])
            l_f_meta = (cost1 * eps[:, 1]).mean() + cost2[1]
            meta_net.zero_grad()
            
            grads = torch.autograd.grad(l_f_meta, meta_net.parameters(), create_graph = True)
            meta_net.update_params(0.001, source_params = grads)
            try:
                val_data, val_Y, _, val_labels, val_ids, val_p = next(dataloader_test)
            except StopIteration:
                dataloader_test = iter(test_loader)
                val_data, val_Y, _, val_labels, val_ids, val_p = next(dataloader_test)

            val_data = self.to_var(val_data, requires_grad = False)
            if self.config['dataset_name'] == 'mnist':
                val_data = val_data.view(-1, 784)
            val_labels = self.to_var(val_labels, requires_grad=False).float()
            y_g_hat = meta_net(val_data)
            
            val_prob = torch.sigmoid(y_g_hat)
            val_prob = torch.cat([1 - val_prob, val_prob], dim=1)

            l_g_meta = -torch.mean(torch.sum(val_prob * torch.log(val_prob + 1e-10), dim = 1)) * 2
            
            grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0] 
            #print(grad_eps) 
            w = torch.clamp(-grad_eps, min = 0)
            w[:, 0] = w[:, 0] + 1e-10
            acount = 0
            bcount = 0
            ccount = 0
            dcount = 0

            for j in range(w.shape[0]):
                if Y[j] == -1:
                    if torch.sum(w[:, 1]) >= self.config['gamma'] * self.config['batch_size']['train']:
                        w[j, 0] = 1
                        w[j, 1] = 0
                    else:
                        w[j, :] = w[j, :] / torch.sum(w[j, :])
                else:
                    w[j, 0] = 1
                    w[j, 1] = 0
                w = w.to(self.config['device']).detach()
            # compute output

            output1 = model1(X)
            output2 = model2(X)
            with torch.no_grad():
                ema_output2 = ema_model2(X)

            _, loss = criterion(output2, Y, eps = w[:, 0])
            consistency_loss = consistency_weight * \
            consistency_criterion(output2, ema_output2) / X.shape[0]
            #print(loss2)
            predictions2 = torch.sign(output2).long()
            predictiont2 = torch.sign(ema_output2).long()

            smx1 = torch.sigmoid(output1) # 计算sigmoid概率
            smx1 = torch.cat([1 - smx1, smx1], dim=1) # 组合成预测变量

            smxY = ((Y + 1) // 2).long() # 分类结果，0-1分类

            smx2 = torch.sigmoid(output2) # 计算sigmoid概率
            smx2 = torch.cat([1 - smx2, smx2], dim=1) # 组合成预测变量

            xent = -torch.sum(smx2 * torch.log(smx2 + 1e-10), dim = 1)

            if self.config['type'] == 'mu' and self.check_mean_teacher(epoch):
                aux = torch.nn.functional.mse_loss(smx2[:, 0], smx1[:, 0].detach())
                if aux < loss * self.config['alpha']:
                    loss += aux
                    count_noisy.update(1, X.size(0))
                else:
                    count_noisy.update(0, X.size(0))
            elif self.config['type'] == 'ori':
                pass

            if self.check_mean_teacher(epoch):
                loss += consistency_loss

            loss += (xent * w[:, 1]).mean()
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

            pacc_2, nacc_2, pnacc_2, psize = accuracy(predictions2, T)
            pacc_4, nacc_4, pnacc_4, psize = accuracy(predictiont2, T) 
            pacc2.update(pacc_2, psize)
            nacc2.update(nacc_2, X.size(0) - psize)
            pnacc2.update(pnacc_2, X.size(0))
            pacc4.update(pacc_4, psize)
            nacc4.update(nacc_4, X.size(0) - psize)
            pnacc4.update(pnacc_4, X.size(0))
            w2.update(torch.sum(w[:, 0]).item(), 1)
            ########################################################################
            # calculating score-related values
            ########################################################################

            f_x = output1
            detached_f_x = f_x.clone().detach().cpu()
            y_batch_prob = self.predict_prob_y_given_X(f_x=detached_f_x)
            l_batch_prob = self.predict_proba(f_x=detached_f_x)
            y_batch_est = self.predict_y_given_X(f_x=detached_f_x)
            l_batch_est = self.predict(f_x=detached_f_x)

            if isinstance(y_batch_prob, np.ndarray):
                y_batch_prob = torch.tensor(y_batch_prob, dtype=LPU.constants.DTYPE)
                l_batch_prob = torch.tensor(l_batch_prob, dtype=LPU.constants.DTYPE)
                y_batch_est = torch.tensor(y_batch_est, dtype=LPU.constants.DTYPE)
                l_batch_est = torch.tensor(l_batch_est, dtype=LPU.constants.DTYPE)

            loss_batch_concat.append(loss.item())
            l_batch_concat.append(Y)
            y_batch_concat.append(T)
            y_batch_concat_prob.append(y_batch_prob)
            l_batch_concat_prob.append(l_batch_prob)
            l_batch_concat_est.append(l_batch_est)
            y_batch_concat_est.append(y_batch_est)  


        if self.check_mean_teacher(epoch):
            self.update_ema_variables(model1, ema_model1,) # 更新ema参数
            self.update_ema_variables(model2, ema_model2,)
            self.mean_teacher_step += 1


        LOG.info('Epoch Noisy : [{0}]\t'
                'PACC1 {pacc1.val:.3f} ({pacc1.avg:.3f})\t'
                'NACC1 {nacc1.val:.3f} ({nacc1.avg:.3f})\t'
                'PNACC1 {pnacc1.val:.3f} ({pnacc1.avg:.3f})\t'
                'PACC2 {pacc2.val:.3f} ({pacc2.avg:.3f})\t'
                'NACC2 {nacc2.val:.3f} ({nacc2.avg:.3f})\t'
                'PNACC2 {pnacc2.val:.3f} ({pnacc2.avg:.3f})\t'
                'PACC3 {pacc3.val:.3f} ({pacc3.avg:.3f})\t'
                'NACC3 {nacc3.val:.3f} ({nacc3.avg:.3f})\t'
                'PNACC3 {pnacc3.val:.3f} ({pnacc3.avg:.3f})\t'
                'PACC4 {pacc4.val:.3f} ({pacc4.avg:.3f})\t'
                'NACC4 {nacc4.val:.3f} ({nacc4.avg:.3f})\t'
                'PNACC4 {pnacc4.val:.3f} ({pnacc4.avg:.3f})\t'
                'W1 ({w1.avg:.3f})\t'
                'W2 ({w2.avg:.3f})\t'.format(
                epoch, pacc1=pacc1, nacc1=nacc1, pnacc1=pnacc1, 
                pacc2=pacc2, nacc2=nacc2, pnacc2=pnacc2, pacc3=pacc3, nacc3=nacc3, pnacc3=pnacc3, 
                pacc4=pacc4, nacc4=nacc4, pnacc4=pnacc4, w1 = w1, w2 = w2))
        
        y_batch_concat_prob = np.concatenate(y_batch_concat_prob)
        l_batch_concat_prob = np.concatenate(l_batch_concat_prob)
        y_batch_concat_est = np.concatenate(y_batch_concat_est)
        l_batch_concat_est = np.concatenate(l_batch_concat_est)
        y_batch_concat = np.concatenate(y_batch_concat)
        l_batch_concat = np.concatenate(l_batch_concat)
        if binary_kind == {-1, 1}:
            l_batch_concat = (l_batch_concat + 1) // 2
            y_batch_concat = (y_batch_concat + 1) // 2
        scores_dict = self._calculate_validation_metrics(
            y_batch_concat_prob, y_batch_concat, l_batch_concat_prob, l_batch_concat, l_ests=l_batch_concat_est, y_ests=y_batch_concat_est
        )

        # average loss
        scores_dict['overall_loss'] = np.mean(loss_batch_concat)

        return scores_dict          

    def train(self, clean1_loader, noisy1_loader, clean2_loader, noisy2_loader, model1, model2, ema_model1, ema_model2, criterion, consistency_criterion, optimizer1, scheduler1, optimizer2, scheduler2, epoch, warmup = False, self_paced_pick = 0):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        #losses = AverageMeter()
        pacc1 = AverageMeter()
        nacc1 = AverageMeter()
        pnacc1 = AverageMeter()
        pacc2 = AverageMeter()
        nacc2 = AverageMeter()
        pnacc2 = AverageMeter()
        pacc3 = AverageMeter()
        nacc3 = AverageMeter()
        pnacc3 = AverageMeter()
        pacc4 = AverageMeter()
        nacc4 = AverageMeter()
        pnacc4 = AverageMeter()
        count_clean = AverageMeter()
        count_noisy = AverageMeter()
        model1.train()
        model2.train()
        ema_model1.train()
        ema_model2.train()
        end = time.time()

        entropy_clean = AverageMeter()
        entropy_noisy = AverageMeter()

        count2 = AverageMeter()
        count1 = AverageMeter()
        consistency_weight = self.get_current_consistency_weight(epoch - 30)
        resultt = np.zeros(61000)
        
        binary_kind = set(np.unique(noisy1_loader.dataset.Y))

        if clean1_loader: 
            for i, (X, _, Y, T, ids, _) in enumerate(clean1_loader):
            # measure data loading time
            
                data_time.update(time.time() - end)
                X = X.to(self.config['device'])
                if self.config['dataset_name'] == 'mnist':
                    X = X.view(X.shape[0], -1)
                Y = Y.to(self.config['device']).float()
                T = T.to(self.config['device']).long()
                # compute output
                output1 = model1(X)
                output2 = model2(X)
                with torch.no_grad():
                    ema_output1 = ema_model1(X)

                consistency_loss = consistency_weight * \
                consistency_criterion(output1, ema_output1) / X.shape[0]

                predictiont1   = torch.sign(ema_output1).long()
                predictions1 = torch.sign(output1).long() # 否则使用自己的结果

                smx1 = torch.sigmoid(output1) # 计算sigmoid概率
                smx1 = torch.cat([1 - smx1, smx1], dim=1) # 组合成预测变量
                smxY = ((Y + 1) // 2).long() # 分类结果，0-1分类
                smx2 = torch.sigmoid(output2) # 计算sigmoid概率
                smx2 = torch.cat([1 - smx2, smx2], dim=1) # 组合成预测变量

                if self.config['soft_label']:
                    aux1 = - torch.mean(torch.sum(smx1 * torch.log(smx1 + 1e-10), dim = 1))
                    aux2 = - torch.mean(torch.sum(smx2 * torch.log(smx2 + 1e-10), dim = 1))
                else:
                    smxY = smxY.float()
                    smxY = smxY.view(-1, 1)
                    smxY = torch.cat([1 - smxY, smxY], dim = 1)
                    aux1 = - torch.mean(torch.sum(smxY * torch.log(smx1 + 1e-10), dim = 1)) # 计算Xent loss
                    aux2 = - torch.mean(torch.sum(smxY * torch.log(smx2 + 1e-10), dim = 1)) # 计算Xent loss

                
                loss = aux1
                entropy_clean.update(aux1, 1)

                if self.check_mean_teacher(epoch):
                    loss += consistency_loss

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()

                
            
        if clean2_loader: 
            for i, (X, _, Y, T, ids, _) in enumerate(clean2_loader):
            # measure data loading time
            
                data_time.update(time.time() - end)
                X = X.to(self.config['device'])
                if self.config['dataset_name'] == 'mnist':
                    X = X.view(X.shape[0], -1)
                Y = Y.to(self.config['device']).float()
                T = T.to(self.config['device']).long()
                # compute output
                output1 = model1(X)
                output2 = model2(X)
                with torch.no_grad():
                    ema_output2 = ema_model2(X)

                consistency_loss = consistency_weight * \
                consistency_criterion(output2, ema_output2) / X.shape[0]

                predictiont2 = torch.sign(ema_output2).long()
                predictions2 = torch.sign(output2).long()

                smx1 = torch.sigmoid(output1) # 计算sigmoid概率
                smx1 = torch.cat([1 - smx1, smx1], dim=1) # 组合成预测变量

                smxY = ((Y + 1) // 2).long() # 分类结果，0-1分类

                smx2 = torch.sigmoid(output2) # 计算sigmoid概率
                smx2 = torch.cat([1 - smx2, smx2], dim=1) # 组合成预测变量

                if self.config['soft_label']:
                    aux1 = - torch.mean(torch.sum(smx1 * torch.log(smx1 + 1e-10), dim = 1))
                    aux2 = - torch.mean(torch.sum(smx2 * torch.log(smx2 + 1e-10), dim = 1))
                else:
                    smxY = smxY.float()
                    smxY = smxY.view(-1, 1)
                    smxY = torch.cat([1 - smxY, smxY], dim = 1)
                    aux1 = - torch.mean(torch.sum(smxY * torch.log(smx1 + 1e-10), dim = 1)) # 计算Xent loss
                    aux2 = - torch.mean(torch.sum(smxY * torch.log(smx2 + 1e-10), dim = 1)) # 计算Xent loss

                loss = aux2
                entropy_clean.update(aux2, 1)

                if self.check_mean_teacher(epoch):
                    loss += consistency_loss
                    
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()

                pacc_2, nacc_2, pnacc_2, psize = accuracy(predictions2, T)
                pacc_4, nacc_4, pnacc_4, psize = accuracy(predictiont2, T) 
                pacc2.update(pacc_2, psize)
                nacc2.update(nacc_2, X.size(0) - psize)
                pnacc2.update(pnacc_2, X.size(0))
                pacc4.update(pacc_4, psize)
                nacc4.update(nacc_4, X.size(0) - psize)
                pnacc4.update(pnacc_4, X.size(0))




        if self.check_mean_teacher(epoch):
            self.update_ema_variables(model1, ema_model1) # 更新ema参数
            self.update_ema_variables(model2, ema_model2)
            self.mean_teacher_step += 1

        LOG.info('Epoch Clean : [{0}]\t'
                'PACC1 {pacc1.val:.3f} ({pacc1.avg:.3f})\t'
                'NACC1 {nacc1.val:.3f} ({nacc1.avg:.3f})\t'
                'PNACC1 {pnacc1.val:.3f} ({pnacc1.avg:.3f})\t'
                'PACC2 {pacc2.val:.3f} ({pacc2.avg:.3f})\t'
                'NACC2 {nacc2.val:.3f} ({nacc2.avg:.3f})\t'
                'PNACC2 {pnacc2.val:.3f} ({pnacc2.avg:.3f})\t'
                'PACC3 {pacc3.val:.3f} ({pacc3.avg:.3f})\t'
                'NACC3 {nacc3.val:.3f} ({nacc3.avg:.3f})\t'
                'PNACC3 {pnacc3.val:.3f} ({pnacc3.avg:.3f})\t'
                'PACC4 {pacc4.val:.3f} ({pacc4.avg:.3f})\t'
                'NACC4 {nacc4.val:.3f} ({nacc4.avg:.3f})\t'
                'PNACC4 {pnacc4.val:.3f} ({pnacc4.avg:.3f})\t'.format(
                epoch, pacc1=pacc1, nacc1=nacc1, pnacc1=pnacc1, 
                pacc2=pacc2, nacc2=nacc2, pnacc2=pnacc2, pacc3=pacc3, nacc3=nacc3, pnacc3=pnacc3, 
                pacc4=pacc4, nacc4=nacc4, pnacc4=pnacc4))
        
        #if epoch > self.config['self_paced_start']: criterion.update_p(0.05)
        if (self.config['dataset_name'] == 'cifar'):
            criterion.update_p((20000 - self_paced_pick / 2) / (50000 - self_paced_pick))
            LOG.info("Setting Pi_P to {}".format((20000 - self_paced_pick / 2) / (50000 - self_paced_pick)))

        l_batch_concat = []
        y_batch_concat = []
        y_batch_concat_prob = []
        l_batch_concat_prob = []
        l_batch_concat_est = []
        y_batch_concat_est = []
        loss_batch_concat = []
        for i, (X, _, Y, T, ids, _) in enumerate(noisy1_loader):
            #print(torch.max(X))
            X = X.to(self.config['device'])
            if self.config['dataset_name'] == 'mnist':
                X = X.view(X.shape[0], -1)
            Y = Y.to(self.config['device']).float()
            T = T.to(self.config['device']).long()
            # compute output
            output1 = model1(X)
            output2 = model2(X)

            with torch.no_grad():
                ema_output1 = ema_model1(X)
            #if epoch >= self.config['self_paced_start']: criterion.update_p(0.5)
            _, loss = criterion(output1, Y)
            consistency_loss = consistency_weight * \
            consistency_criterion(output1, ema_output1) / X.shape[0]
            #print(loss1)

            predictions1 = torch.sign(output1).long()
            predictiont1 = torch.sign(ema_output1).long()

            smx1 = torch.sigmoid(output1) # 计算sigmoid概率
            smx1 = torch.cat([1 - smx1, smx1], dim=1) # 组合成预测变量

            smxY = ((Y + 1) // 2).long() # 分类结果，0-1分类

            smx2 = torch.sigmoid(output2) # 计算sigmoid概率
            smx2 = torch.cat([1 - smx2, smx2], dim=1) # 组合成预测变量
            

            if self.config['type'] == 'mu' and self.check_mean_teacher(epoch):
                aux = torch.nn.functional.mse_loss(smx1[:, 0], smx2[:, 0].detach())
                if aux < loss * self.config['alpha']:
                    loss += aux
                    count_noisy.update(1, X.size(0))
                else:
                    count_noisy.update(0, X.size(0))

            if self.check_mean_teacher(epoch):
                loss += consistency_loss
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            pacc_3, nacc_3, pnacc_3, psize = accuracy(predictiont1, T) 
            pacc_1, nacc_1, pnacc_1, psize = accuracy(predictions1, T) # 使用T来计算预测准确率
            pacc1.update(pacc_1, psize)
            nacc1.update(nacc_1, X.size(0) - psize)
            pnacc1.update(pnacc_1, X.size(0))
            pacc3.update(pacc_3, psize)
            nacc3.update(nacc_3, X.size(0) - psize)
            pnacc3.update(pnacc_3, X.size(0))

            ########################################################################
            # calculating score-related values
            ########################################################################

            f_x = output1
            detached_f_x = f_x.clone().detach().cpu()
            y_batch_prob = self.predict_prob_y_given_X(f_x=detached_f_x)
            l_batch_prob = self.predict_proba(f_x=detached_f_x)
            y_batch_est = self.predict_y_given_X(f_x=detached_f_x)
            l_batch_est = self.predict(f_x=detached_f_x)

            if isinstance(y_batch_prob, np.ndarray):
                y_batch_prob = torch.tensor(y_batch_prob, dtype=LPU.constants.DTYPE)
                l_batch_prob = torch.tensor(l_batch_prob, dtype=LPU.constants.DTYPE)
                y_batch_est = torch.tensor(y_batch_est, dtype=LPU.constants.DTYPE)
                l_batch_est = torch.tensor(l_batch_est, dtype=LPU.constants.DTYPE)

            loss_batch_concat.append(loss.item())
            l_batch_concat.append(Y)
            y_batch_concat.append(T)
            y_batch_concat_prob.append(y_batch_prob)
            l_batch_concat_prob.append(l_batch_prob)
            l_batch_concat_est.append(l_batch_est)
            y_batch_concat_est.append(y_batch_est)

        loss_1_average = loss / len(noisy1_loader)
        for i, (X, _, Y, T, ids, _) in enumerate(noisy2_loader):

            X = X.to(self.config['device'])
            if self.config['dataset_name'] == 'mnist':
                X = X.view(X.shape[0], -1)
            Y = Y.to(self.config['device']).float()
            T = T.to(self.config['device']).long()

            # compute output
            output1 = model1(X)
            output2 = model2(X)
            with torch.no_grad():
                ema_output2 = ema_model2(X)

            _, loss = criterion(output2, Y)
            consistency_loss = consistency_weight * \
            consistency_criterion(output2, ema_output2) / X.shape[0]
            #print(loss2)
            predictions2 = torch.sign(output2).long()
            predictiont2 = torch.sign(ema_output2).long()

            smx1 = torch.sigmoid(output1) # 计算sigmoid概率
            smx1 = torch.cat([1 - smx1, smx1], dim=1) # 组合成预测变量

            smxY = ((Y + 1) // 2).long() # 分类结
            smx2 = torch.sigmoid(output2) # 计算sigmoid概率
            smx2 = torch.cat([1 - smx2, smx2], dim=1) # 组合成预测变量


            #aux2 = - torch.sum(smx2 * torch.log(smx2)) / smx2.shape[0]
            #entropy_noisy.update(aux2, 1)

            if self.config['type'] == 'mu' and self.check_mean_teacher(epoch):
                aux = torch.nn.functional.mse_loss(smx2[:, 0], smx1[:, 0].detach())
                if aux < loss * self.config['alpha']:
                    loss += aux
                    count_noisy.update(1, X.size(0))
                else:
                    count_noisy.update(0, X.size(0))

            if self.check_mean_teacher(epoch):
                loss += consistency_loss

            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

            pacc_2, nacc_2, pnacc_2, psize = accuracy(predictions2, T)
            pacc_4, nacc_4, pnacc_4, psize = accuracy(predictiont2, T) 
            pacc2.update(pacc_2, psize)
            nacc2.update(nacc_2, X.size(0) - psize)
            pnacc2.update(pnacc_2, X.size(0))
            pacc4.update(pacc_4, psize)
            nacc4.update(nacc_4, X.size(0) - psize)
            pnacc4.update(pnacc_4, X.size(0))

        if self.check_mean_teacher(epoch):
            self.update_ema_variables(model1, ema_model1) # 更新ema参数
            self.update_ema_variables(model2, ema_model2)
            self.mean_teacher_step += 1

        if not warmup: 
            scheduler1.step()
            scheduler2.step()

        LOG.info(count_clean.avg)
        LOG.info(count_noisy.avg)

        #print(entropy_clean.avg)
        #print(entropy_noisy.avg)

        LOG.info('Epoch Noisy : [{0}]\t'
                'PACC1 {pacc1.val:.3f} ({pacc1.avg:.3f})\t'
                'NACC1 {nacc1.val:.3f} ({nacc1.avg:.3f})\t'
                'PNACC1 {pnacc1.val:.3f} ({pnacc1.avg:.3f})\t'
                'PACC2 {pacc2.val:.3f} ({pacc2.avg:.3f})\t'
                'NACC2 {nacc2.val:.3f} ({nacc2.avg:.3f})\t'
                'PNACC2 {pnacc2.val:.3f} ({pnacc2.avg:.3f})\t'
                'PACC3 {pacc3.val:.3f} ({pacc3.avg:.3f})\t'
                'NACC3 {nacc3.val:.3f} ({nacc3.avg:.3f})\t'
                'PNACC3 {pnacc3.val:.3f} ({pnacc3.avg:.3f})\t'
                'PACC4 {pacc4.val:.3f} ({pacc4.avg:.3f})\t'
                'NACC4 {nacc4.val:.3f} ({nacc4.avg:.3f})\t'
                'PNACC4 {pnacc4.val:.3f} ({pnacc4.avg:.3f})\t'.format(
                epoch, pacc1=pacc1, nacc1=nacc1, pnacc1=pnacc1, 
                pacc2=pacc2, nacc2=nacc2, pnacc2=pnacc2, pacc3=pacc3, nacc3=nacc3, pnacc3=pnacc3, 
                pacc4=pacc4, nacc4=nacc4, pnacc4=pnacc4))

        y_batch_concat_prob = np.concatenate(y_batch_concat_prob)
        l_batch_concat_prob = np.concatenate(l_batch_concat_prob)
        y_batch_concat_est = np.concatenate(y_batch_concat_est)
        l_batch_concat_est = np.concatenate(l_batch_concat_est)
        y_batch_concat = np.concatenate(y_batch_concat)
        l_batch_concat = np.concatenate(l_batch_concat)
        if binary_kind == {-1, 1}:
            l_batch_concat = (l_batch_concat + 1) // 2
            y_batch_concat = (y_batch_concat + 1) // 2
        scores_dict = self._calculate_validation_metrics(
            y_batch_concat_prob, y_batch_concat, l_batch_concat_prob, l_batch_concat, l_ests=l_batch_concat_est, y_ests=y_batch_concat_est
        )

        # average loss
        scores_dict['overall_loss'] = np.mean(loss_batch_concat)

        return scores_dict


    def predict_prob_y_given_X(self, X=None, f_x=None):
        if f_x is None:
            f_x = self.model(X)
        return torch.sigmoid(f_x)

    def predict_prob_l_given_y_X(self, X=None, f_x=None):
        return 1.

    def validate(self, dataloader, loss_fn=None, model=None):
        scores_dict = {}
        total_loss = 0.
        l_batch_concat = []
        y_batch_concat = []
        y_batch_concat_prob = []
        l_batch_concat_prob = []
        l_batch_concat_est = []
        y_batch_concat_est = []
        model.eval()
        binary_kind = set(np.unique(dataloader.dataset.y if hasattr(dataloader.dataset, 'y') else dataloader.dataset.Y))
        with torch.no_grad():
            for batch_num, (X_batch, _, l_batch, y_batch, ids, _) in enumerate(dataloader):

                if self.config['data_generating_process'] == 'CC':
                    X_batch = torch.concat([X_batch, X_batch[l_batch==1]], dim=0)
                    y_batch = torch.concat([y_batch, y_batch[l_batch==1]], dim=0)
                    l_batch = torch.concat([torch.zeros_like(l_batch), torch.ones((int(l_batch.sum().detach().cpu().numpy().squeeze())))], dim=0)
                f_x = model(X_batch)
                _, loss = loss_fn(f_x, l_batch)
                y_batch_prob = self.predict_prob_y_given_X(f_x=f_x)
                l_batch_prob = self.predict_proba(f_x=f_x)
                y_batch_est = self.predict_y_given_X(f_x=f_x)
                l_batch_est = self.predict(f_x=f_x)
                
                if isinstance(y_batch_prob, np.ndarray):
                    y_batch_prob = torch.tensor(y_batch_prob, dtype=LPU.constants.DTYPE)
                    l_batch_prob = torch.tensor(l_batch_prob, dtype=LPU.constants.DTYPE)
                    y_batch_est = torch.tensor(y_batch_est, dtype=LPU.constants.DTYPE)
                    l_batch_est = torch.tensor(l_batch_est, dtype=LPU.constants.DTYPE)


                total_loss += loss.item()

                l_batch_concat.append(l_batch.detach().cpu().numpy())
                y_batch_concat.append(y_batch.detach().cpu().numpy())
                y_batch_concat_prob.append(y_batch_prob.detach().cpu().numpy())
                l_batch_concat_prob.append(l_batch_prob.detach().cpu().numpy())
                y_batch_concat_est.append(y_batch_est.detach().cpu().numpy())
                l_batch_concat_est.append(l_batch_est.detach().cpu().numpy())   

        y_batch_concat_prob = np.concatenate(y_batch_concat_prob)
        l_batch_concat_prob = np.concatenate(l_batch_concat_prob)
        y_batch_concat_est = np.concatenate(y_batch_concat_est)
        l_batch_concat_est = np.concatenate(l_batch_concat_est)
        y_batch_concat = np.concatenate(y_batch_concat)
        l_batch_concat = np.concatenate(l_batch_concat)

        if binary_kind == {-1, 1}:
            y_batch_concat = (y_batch_concat + 1) / 2
            l_batch_concat = (l_batch_concat + 1) / 2
        scores_dict = self._calculate_validation_metrics(
            y_batch_concat_prob, y_batch_concat, l_batch_concat_prob, l_batch_concat, l_ests=l_batch_concat_est, y_ests=y_batch_concat_est
        )

        # for score_type in scores_dict:
        #     scores_dict[score_type] = np.mean(scores_dict[score_type])
        total_loss /= (batch_num + 1)
        scores_dict['overall_loss'] = total_loss

        return scores_dict    
    
    def update_dataset(self, model1, model2, ema_model1, ema_model2, dataset_train1_clean, dataset_train1_noisy, dataset_train2_clean, dataset_train2_noisy, epoch, ratio=0.5):
        #global results
        dataset_train1_noisy.reset_ids()
        dataset_train1_noisy.set_type("clean")
        dataset_train2_noisy.reset_ids()
        dataset_train2_noisy.set_type("clean")
        dataloader_train1 = torch.utils.data.DataLoader(dataset_train1_noisy, batch_size=self.config['batch_size']['train'], num_workers=self.config['workers'], shuffle=False, pin_memory=True)
        dataloader_train2 = torch.utils.data.DataLoader(dataset_train2_noisy, batch_size=self.config['batch_size']['train'], num_workers=self.config['workers'], shuffle=False, pin_memory=True)
        results1 = np.zeros(len(dataset_train1_noisy))# + self.config['num_p'])
        results2 = np.zeros(len(dataset_train1_noisy))# + self.config['num_p'])
        model1.eval()
        model2.eval()
        # validation #######################
        with torch.no_grad():
            for i, (X, _, Y, T, ids, _) in enumerate(tqdm.tqdm(dataloader_train1)):
                #images, lefts, rights, ages, genders, edus, apoes, labels, pu_labels, ids = Variable(sample_batched['mri']).to('cpu'), Variable(sample_batched['left']).to('cpu'), Variable(sample_batched['right']).to('cpu'), Variable(sample_batched['age']).to('cpu'), Variable(sample_batched['gender']).to('cpu'), Variable(sample_batched['edu']).to('cpu'), Variable(sample_batched['apoe']).to('cpu'), Variable(sample_batched['label']).view(-1).type(torch.LongTensor).to('cpu'), Variable(sample_batched['pu_label']).view(-1).type(torch.LongTensor).to('cpu'), sample_batched['id']
                X = X.to('cpu')
                Y = Y.to('cpu')
                Y = Y.float()
                # ===================forward====================
                if self.check_mean_teacher(epoch):
                    output1 = ema_model1(X)
                else:
                    output1 = model1(X)
                prob1 = torch.sigmoid(output1).view(-1).cpu().numpy()       
                results1[ids.view(-1).numpy()] = prob1

            for i, (X, _, Y, T, ids, _) in enumerate(tqdm.tqdm(dataloader_train2)):
                #images, lefts, rights, ages, genders, edus, apoes, labels, pu_labels, ids = Variable(sample_batched['mri']).to('cpu'), Variable(sample_batched['left']).to('cpu'), Variable(sample_batched['right']).to('cpu'), Variable(sample_batched['age']).to('cpu'), Variable(sample_batched['gender']).to('cpu'), Variable(sample_batched['edu']).to('cpu'), Variable(sample_batched['apoe']).to('cpu'), Variable(sample_batched['label']).view(-1).type(torch.LongTensor).to('cpu'), Variable(sample_batched['pu_label']).view(-1).type(torch.LongTensor).to('cpu'), sample_batched['id']
                X = X.to('cpu')
                Y = Y.to('cpu')
                Y = Y.float()
                # ===================forward====================
                if self.check_mean_teacher(epoch):
                    output2 = ema_model2(X)
                else:
                    output2 = model2(X)
                prob2 = torch.sigmoid(output2).view(-1).cpu().numpy()     
                results2[ids.view(-1).numpy()] = prob2

        # adni_dataset_train.update_labels(results, ratio)
        # dataset_origin = dataset_train
        ids_noisy1 = dataset_train1_clean.update_ids(results1, epoch, ratio = ratio) # 返回的是noisy ids
        ids_noisy2 = dataset_train2_clean.update_ids(results2, epoch, ratio = ratio)
        dataset_train1_noisy.set_ids(ids_noisy1) # 将noisy ids更新进去
        dataset_train1_noisy.set_type("noisy")
        dataset_train2_noisy.set_ids(ids_noisy2) # 将noisy ids更新进去
        dataset_train2_noisy.set_type("noisy")
        dataset_train1_clean.set_type("clean")
        dataset_train2_clean.set_type("clean")

        #assert np.all(dataset_train_noisy.ids == ids_noisy) # 确定更新了
        #dataloader_origin = torch.utils.data.DataLoader(dataset_origin, batch_size =self.config['batch_size']['train'], num_workers=4, drop_last=True, shuffle=True, pin_memory=True)
        dataloader_train1_clean = torch.utils.data.DataLoader(dataset_train1_clean, batch_size =self.config['batch_size']['train'], num_workers=self.config['workers'], shuffle=True, pin_memory=True)
        dataloader_train1_noisy = torch.utils.data.DataLoader(dataset_train1_noisy, batch_size =self.config['batch_size']['train'], num_workers=self.config['workers'], shuffle=False, pin_memory=True)
        dataloader_train2_clean = torch.utils.data.DataLoader(dataset_train2_clean, batch_size =self.config['batch_size']['train'], num_workers=self.config['workers'], shuffle=True, pin_memory=True)
        dataloader_train2_noisy = torch.utils.data.DataLoader(dataset_train2_noisy, batch_size =self.config['batch_size']['train'], num_workers=self.config['workers'], shuffle=False, pin_memory=True)
        return dataloader_train1_clean, dataloader_train1_noisy, dataloader_train2_clean, dataloader_train2_noisy    