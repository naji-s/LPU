import torch
import torch.utils.data
import torchvision.transforms
import numpy as np

import lpu.external_libs.PU_learning
import lpu.external_libs.PU_learning.data
import lpu.external_libs.PU_learning.data_helper
import lpu.external_libs.PU_learning.helper
import lpu.external_libs.PU_learning.model_helper
import lpu.external_libs.PU_learning.data_helper.IMDb

from torch.utils.data import DataLoader, SubsetRandomSampler

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = dataset.targets[indices]  # Subset of targets

        # Find indices for specific conditions within the subset
        # self.p_data_idx = np.where(self.targets == 1)[0]
        # self.n_data_idx = np.where(self.targets == 7)[0]
        self.target_transform = target_transform    
        # Store the subset data corresponding to specific conditions
        self.p_data = np.asarray([self.__getitem__(idx)[0] for idx in self.indices])
        self.n_data = np.asarray([self.__getitem__(idx)[0] for idx in self.indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, target = self.dataset[original_idx]
        img = img.detach().cpu().numpy()
        if self.transform:
            img = self.transform(img)
        return img, target

    
# Custom DataLoader that applies transformation
def get_loader(indices, dataset, transform):
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = [transform(image) for image in images]  # Apply transformation
        return torch.stack(images), torch.tensor(labels)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetRandomSampler(indices),
        collate_fn=collate_fn
    )


def get_dataset(data_dir, data_type,net_type, device, alpha, beta, batch_size, config=None): 
    if config:
        train_ratio = config['ratios']['train']
        val_ratio = config['ratios']['val']
        test_ratio = config['ratios']['test']
    if train_ratio + val_ratio  + test_ratio < 1.0:
        raise ValueError("Train + Val + Test ratio should be 1.0")
    
    p_trainloader=None
    u_trainloader=None
    p_validloader=None
    u_validloader=None
    p_testloader=None
    u_testloader=None

    net=None
    X=None
    Y=None

    if data_type=='gaussian': 
        '''
        Gaussian Data hyperparamters and data
        '''
        num_points = 6000
        input_size = 200
        # dividing by train_ratio to get the number of positive points in the training set
        # that is in the original repo, the number of positive points is 2000
        pos_size = 2000 / train_ratio
        train_size = int(num_points * train_ratio)
        val_size = int(num_points * val_ratio)
        test_size = num_points - train_size - val_size
        train_pos_size = int(pos_size * train_ratio)
        val_pos_size = int(pos_size * val_ratio)
        test_pos_size = pos_size - train_pos_size - val_pos_size


        gauss_traindata = lpu.external_libs.PU_learning.data_helper.toy.Gaussain_data(mu=1.0, sigma=np.sqrt(input_size//2), size=train_size, dim=input_size//2)
        gauss_validdata = lpu.external_libs.PU_learning.data_helper.toy.Gaussain_data(mu=1.0, sigma=np.sqrt(input_size//2), size=val_size, dim=input_size//2)
        gauss_testdata = lpu.external_libs.PU_learning.data_helper.toy.Gaussain_data(mu=1.0, sigma=np.sqrt(input_size//2), size=test_size, dim=input_size//2)


        p_traindata, u_traindata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(gauss_traindata, pos_size=train_pos_size, alpha=alpha, beta=beta)
        p_validdata, u_validdata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(gauss_validdata, pos_size=val_pos_size, alpha=alpha, beta=beta)
        p_testdata, u_testdata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(gauss_testdata, pos_size=test_pos_size, alpha=alpha, beta=beta)

        X = p_traindata.data
        Y = u_traindata.data

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_testloader = torch.utils.data.DataLoader(p_testdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_testloader = torch.utils.data.DataLoader(u_testdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 

        net = lpu.external_libs.PU_learning.model_helper.get_model(net_type, input_dim = input_size)
        net = net.to(device)

    elif data_type=='toy_continuous':       
        '''
        Toy dataset from P vs U failure for domain discrimination
        '''

        toy_traindata = lpu.external_libs.PU_learning.data_helper.toy.ToyDataContinuous()
        toy_validdata = lpu.external_libs.PU_learning.data_helper.toy.ToyDataContinuous()
        toy_testdata = lpu.external_libs.PU_learning.data_helper.toy.ToyDataContinuous()

        # dividing by train_ratio to get the number of positive points in the training set
        # that is in the original repo, the number of positive points is 50
        pos_size = 50 / train_ratio

        train_pos_size = int(pos_size * train_ratio)
        val_pos_size = int(pos_size * val_ratio)
        test_pos_size = pos_size - train_pos_size - val_pos_size

        p_traindata, u_traindata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(toy_traindata, pos_size=train_pos_size, alpha=alpha, beta=beta)
        p_validdata, u_validdata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(toy_validdata, pos_size=val_pos_size, alpha=alpha, beta=beta)
        p_testdata, u_testdata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(toy_testdata, pos_size=test_pos_size, alpha=alpha, beta=beta)

        X = p_traindata.data
        Y = u_traindata.data

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        p_testloader = torch.utils.data.DataLoader(p_testdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_testloader = torch.utils.data.DataLoader(u_testdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        

        ## Initialize model 
        net = lpu.external_libs.PU_learning.model_helper.get_model(net_type, input_dim = 2)
        net = net.to(device)

    elif data_type=='toy_discrete': 

        toy_traindata =lpu.external_libs.PU_learning.data_helper.toy.ToyData()
        toy_validdata = lpu.external_libs.PU_learning.data_helper.toy.ToyData()
        toy_testdata = lpu.external_libs.PU_learning.data_helper.toy.ToyData()

        # dividing by train_ratio to get the number of positive points in the training set
        # that is in the original repo, the number of positive points is 8
        pos_size = 8 / train_ratio
        train_pos_size = int(pos_size * train_ratio)
        val_pos_size = int(pos_size * val_ratio)
        test_pos_size = pos_size - train_pos_size - val_pos_size

        p_traindata, u_traindata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(toy_traindata, pos_size=pos_size, alpha=alpha, beta=beta)
        p_validdata, u_validdata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(toy_validdata, pos_size=pos_size, alpha=alpha, beta=beta)
        p_testdata, u_testdata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(toy_testdata, pos_size=pos_size, alpha=alpha, beta=beta)

        X = p_traindata.data
        Y = u_traindata.data

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        p_testloader = torch.utils.data.DataLoader(p_testdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_testloader = torch.utils.data.DataLoader(u_testdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        

        ## Initialize model 
        net = lpu.external_libs.PU_learning.model_helper.get_model(net_type, input_dim = 2)
        net = net.to(device)


    elif data_type=='mnist_17': 

        transform_train = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])


        transform_test = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])
        train_ratio = train_ratio / train_ratio + val_ratio
        val_ratio  = 1 - train_ratio



        traindata = lpu.external_libs.PU_learning.data_helper.MNIST17Data(root=data_dir, train=True, transform=transform_train)
        # Split indices for training and validation
        total_size = len(traindata)
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size
        train_indices, val_indices = torch.utils.data.random_split(range(len(traindata)), [train_size, val_size])

        traindata = SubsetDataset(traindata, train_indices, transform=transform_train)
        validdata = SubsetDataset(traindata, val_indices, transform=transform_test)

        train_pos_size = int(3000 * train_ratio)
        val_pos_size = 3000 - train_pos_size

        testdata = lpu.external_libs.PU_learning.data_helper.MNIST.MNIST17Data(root=data_dir, train=False, transform=transform_test)

        p_traindata, u_traindata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(traindata, pos_size=train_pos_size, alpha=alpha, beta=beta,data_type='mnist')
        p_validdata, u_validdata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(validdata, pos_size=val_pos_size, alpha=alpha, beta=beta,data_type='mnist')
        p_testdata, u_testdata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(testdata, pos_size=500, alpha=alpha, beta=beta,data_type='mnist')


        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_testloader = torch.utils.data.DataLoader(p_testdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_testloader = torch.utils.data.DataLoader(u_testdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        

        ## Initialize model 
        net = lpu.external_libs.PU_learning.model_helper.get_model(net_type, input_dim = 784)
        net = net.to(device)


    elif data_type=='mnist_binarized': 

        transform_train = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])
        transform_test = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])
        train_ratio = train_ratio / train_ratio + val_ratio
        val_ratio  = 1 - train_ratio

        traindata = lpu.external_libs.PU_learning.data_helper.MNIST.BinarizedMNISTData(root=data_dir, train=True, transform=transform_train)
        train_size = int(len(traindata) * train_ratio)
        valid_size = int(len(traindata) * val_ratio)
        traindata, validdata = torch.utils.data.random_split(traindata, [train_size, valid_size])
        
        testdata = lpu.external_libs.PU_learning.data_helper.MNIST.BinarizedMNISTData(root=data_dir, train=False, transform=transform_test)

        p_traindata, u_traindata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(traindata, pos_size=15000, alpha=alpha, beta=beta,data_type='mnist')
        p_validdata, u_validdata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(validdata, pos_size=2500, alpha=alpha, beta=beta,data_type='mnist')
        p_testdata, u_testdata = lpu.external_libs.PU_learning.helper.get_PUDataSplits(testdata, pos_size=2500, alpha=alpha, beta=beta,data_type='mnist')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_testloader = torch.utils.data.DataLoader(p_testdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_testloader = torch.utils.data.DataLoader(u_testdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
                                        

        ## Initialize model 
        net = lpu.external_libs.PU_learning.model_helper.get_model(net_type, input_dim = 784)
        net = net.to(device)


    return (p_trainloader, u_trainloader, 
            p_validloader, u_validloader, 
            p_testloader, u_testloader,
            net, X, Y, 
            p_traindata, u_traindata,
            p_validdata, u_validdata,
            p_testdata, u_testdata 
    )

