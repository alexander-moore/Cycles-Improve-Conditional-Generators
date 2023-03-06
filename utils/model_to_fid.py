import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from fjd_master.fjd_metric import FJDMetric
from fjd_master.embeddings import OneHotEmbedding, InceptionEmbedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audit = False

def one_hot_embedding(labels):
    #y = torch.eye(num_classes)
    #return y[labels]
    #return torch.nn.functional.one_hot(labels)[:,1:]
    
    labels = torch.nn.functional.one_hot(torch.tensor(labels).to(torch.int64), num_classes = c_dim)
    return torch.squeeze(labels)

def get_dataloaders():
    #transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), 
    #                          std=(0.5, 0.5, 0.5))])
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], 
                              std=[0.5])])

    train_set = CIFAR10(root='./data',
                        train=True,
                        download=True,
                        transform=transform)

    test_set = CIFAR10(root='./data',
                       train=False,
                       download=True,
                       transform=transform)

    train_loader = DataLoader(train_set,
                              batch_size=128,
                              shuffle=True,
                              drop_last=True)

    test_loader = DataLoader(test_set,
                             batch_size=128,
                             shuffle=True,
                             drop_last=True)
    

    return train_loader, test_loader

#print_g_sample()
c_dim = 10
v_dim = 100
class GANWrapper:
    def __init__(self, model, conditioning):
        self.model = model
        self.conditioning = conditioning

    def __call__(self, y):
        
        y_nat = y
        if audit == True:
            print(y[0:9])

        batch_size = y.size(0)
        onehot_embedding = OneHotEmbedding(num_classes=10)

        y = onehot_embedding(y).to(device).view(batch_size, c_dim, 1, 1)

        z = torch.randn((batch_size, v_dim, 1, 1), device = device)
        
        if self.conditioning == True:
            samples = self.model(z, y)
        else:
            samples = self.model(z)
        
        if audit == True:
            for i in range(9):
                ind = y_nat[i]
                plt.subplot(330 + 1 + i)
                # plot raw pixel data
                element = 0.5*(samples[i,:].permute(1,2,0).cpu() + 1)
                plt.imshow(element, cmap = 'gray')
            plt.show()
        
        return samples

def get_fid_fjd(G, conditioning = True, samples_per_condition = 10):
    # Model should be
    # G = model_arch()
    # G.load_satate_dict(torch.load('path.pt'))
    v_dim = 100
    z_dim = 100
    c_dim = 10

    train_loader, test_loader = get_dataloaders()
    inception_embedding = InceptionEmbedding(parallel=False)
    onehot_embedding = OneHotEmbedding(num_classes=10)

    gan = GANWrapper(G, conditioning)

    fjd_metric = FJDMetric(gan=gan,
                       reference_loader=train_loader, #cifar10 train
                       condition_loader=test_loader, #cifar10 test
                       image_embedding=inception_embedding, #dont change
                       condition_embedding=onehot_embedding,
                       reference_stats_path='datasets/cifar_train_stats.npz',
                       save_reference_stats=True,
                       samples_per_condition=samples_per_condition, #10
                       cuda=True)

    fid = fjd_metric.get_fid()
    fjd = fjd_metric.get_fjd()
    print('FID: ', fid)
    print('FJD: ', fjd)

    return fid, fjd