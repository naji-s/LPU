import logging
import numpy as np
import torch.nn
import torchvision.models
import LPU.external_libs
import LPU.external_libs.distPU.customized
import LPU.external_libs.distPU.customized.mixup
import LPU.external_libs.distPU.losses
import LPU.external_libs.distPU.losses.distributionLoss
import LPU.external_libs.distPU.losses.entropyMinimization
import LPU.external_libs.distPU.losses.factory
import LPU.external_libs.distPU.models.modelForCIFAR10
import LPU.external_libs.distPU.models.modelForFMNIST
import LPU.models.distPU
import LPU.models.lpu_model_base
import LPU.constants
import LPU.external_libs.distPU.models.factory
import LPU.external_libs.distPU.models.modelForCIFAR10
import LPU.external_libs.distPU.models.modelForFMNIST
import LPU.utils.auxiliary_models

LOG = logging.getLogger(__name__)

class MixupDataset():
    def __init__(self) -> None:
        self.psudo_labels = None
        pass

    def update_psudos(self, data_loader, model, device):
        self.indexes, self.psudo_labels = get_predicted_scores(data_loader, model, device)

def get_predicted_scores(data_loader, model, device):
    model.eval()
    predicted_scores = []
    indexes = []
    with torch.no_grad():
        for epoch, (X, l, _, index) in enumerate(data_loader):
            X = X.to(device)
            l = l.to(device)
            outputs = model(X).squeeze()
            outputs = torch.sigmoid(outputs)
            predicted_scores.append(outputs)
            indexes.append(index.squeeze())
    predicted_scores = torch.cat(predicted_scores)
    indexes = torch.cat(indexes)
    return indexes, predicted_scores

CLASS_PRIOR = {
    'cifar-10': 0.4,
    'fmnist': 0.4,
    'alzheimer': 0.5
}

def create_loss(config, prior=None):
    if prior is None:
        prior = CLASS_PRIOR[config['dataset']]
    print('prior: {}'.format(prior))
    if config['loss'] == 'Dist-PU':
        base_loss = LPU.external_libs.distPU.losses.distributionLoss.LabelDistributionLoss(prior=prior, device=config['device'])
    else:
        raise NotImplementedError("The loss: {} is not defined!".format(args.loss))

    def loss_fn_entropy(outputs, labels):
        scores = torch.sigmoid(torch.clamp(outputs, min=-10, max=10))
        return base_loss(outputs, labels) + config['co_mu'] * LPU.external_libs.distPU.losses.entropyMinimization.loss_entropy(scores[labels!=1])

    if config['entropy'] == 1:
        return loss_fn_entropy
    return base_loss

def create_model(dataset, dim):
    if dataset.startswith('cifar'):
        return LPU.external_libs.distPU.models.modelForCIFAR10.CNN()
    elif dataset.startswith('fmnist'):
        return LPU.external_libs.distPU.models.modelForFMNIST.MultiLayerPerceptron(dim)
    elif dataset == 'alzheimer':
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(2048, 1)
        return model
    else:
        raise NotImplementedError("The model: {} is not defined!".format(dataset))

class distPU(LPU.models.lpu_model_base.LPUModelBase):
    def __init__(self, config, dim):
        super(distPU, self).__init__()
        self.config = config
        self.device = config.get('device')
        self.dim = dim
        # initializing the model
        if config['dataset_kind'] == 'LPU':
            self.model = LPU.utils.auxiliary_models.MultiLayerPerceptron(input_dim=dim, output_dim=1).to(self.device).to(LPU.constants.DTYPE)
        else:
            self.model = LPU.models.distPU.create_model('fmnist', dim=dim).to(self.device).to(LPU.constants.DTYPE)

    def train_one_epoch(self, epoch, dataloader, loss_fn, optimizer, scheduler):
        self.model.train()
        loss_total = 0
        for _, (X, l, _, _) in enumerate(dataloader):
            X = X.to(self.device)
            l = l.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(X).squeeze()
            # in case softmax is the last layer instead of sigmoid
            if outputs.dim() == 2:
                outputs = outputs[:, 1]
            loss = loss_fn(outputs, l.float())
            loss.backward()
            optimizer.step()
            loss_total = loss_total + loss.item()
        scheduler.step()
        return {'overall_loss': loss_total / len(dataloader)}

    def train_mixup_one_epoch(self, epoch, dataloader, loss_fn, optimizer, scheduler, mixup_dataset, co_entropy):
        self.model.train()
        loss_total = 0
        for batch_idx, (X, l, _, index) in enumerate(dataloader):
            X = X.to(self.device)
            l = l.to(self.device)
            psudos = mixup_dataset.psudo_labels[index].to(self.device)
            psudos[l==1] = 1
            mixed_x, y_a, y_b, lam = LPU.external_libs.distPU.customized.mixup.mixup_two_targets(X, psudos, self.config['alpha'], self.device)
            outputs = self.model(mixed_x).squeeze()
            outputs = torch.clamp(outputs, min=-10, max=10)
            scores = torch.sigmoid(outputs)
            outputs_ = torch.clamp(self.model(X).squeeze(), min=-10, max=10).squeeze()
            if outputs_.dim() == 2:
                outputs_ = outputs_[:, 1]
            scores_ = torch.sigmoid(outputs_)
            loss = (
                loss_fn(outputs_, l.float()) +
                co_entropy*LPU.external_libs.distPU.losses.entropyMinimization.loss_entropy(scores_[l!=1]) +
                self.config['co_mix_entropy']*LPU.external_libs.distPU.losses.entropyMinimization.loss_entropy(scores) +
                self.config['co_mixup'] * LPU.external_libs.distPU.customized.mixup.mixup_bce(scores, y_a, y_b, lam)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                mixup_dataset.psudo_labels[index] = scores_.detach()
            loss_total = loss_total + loss.item()
        scheduler.step()
        return {'overall_loss': loss_total / len(dataloader)}

    def predict_prob_y_given_X(self, X=None, f_x=None):
        with torch.no_grad():
            if f_x is None:
                if type(X) == np.ndarray:
                    X = torch.tensor(X, dtype=LPU.constants.DTYPE)
                X = X.to(self.device)
                f_x = self.model(X).squeeze()
            if f_x.flatten().dim() == 2:
                f_x = f_x[:, 1]
            predicted_prob = torch.nn.functional.sigmoid(f_x)
        return predicted_prob.cpu().numpy().squeeze()

    def predict_prob_l_given_y_X(self, X=None, f_x=None):
        return self.C

    def set_C(self, holdout_dataloader):
        X_all = []
        y_all = []
        l_all = []
        for _, (X, l, y, _) in enumerate(holdout_dataloader):
            X_all.append(X)
            y_all.append(y)
            l_all.append(l)
        X_all = torch.cat(X_all)
        y_all = torch.cat(y_all)
        l_all = torch.cat(l_all)
        self.C = l_all[y_all == 1].mean().detach().cpu().numpy()
        self.prior = y_all.mean().detach().cpu().numpy()