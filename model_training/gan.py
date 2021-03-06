def train_gan(exp_num, n_epochs, input_trainset, print_samp = True):    
    # -*- coding: utf-8 -*-
    """GAN_CIFAR10_Final.ipynb

    Automatically generated by Colaboratory.

    Original file is located at
        https://colab.research.google.com/drive/1jzufq2Ia12bIHWrKyWz3lLsecxwcnfx3
    """

    import matplotlib.pyplot as plt
    import itertools
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import tensorflow_datasets as tfds
    import torchvision
    from tensorflow.keras.datasets import cifar10

    from torch.autograd import Variable
    from torch.utils.data.dataset import Dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    ## All Hyperparams should go here. CGAN, CAEGAN, ICAEGAN
    img_size = 32 # can use this to mofidy data size to fit this model
    n_epochs = n_epochs # TESTING VALUE
    print_stride = 1
    bs = 16 # 64

    z_dim = 100
    c_dim = 10

    learning_rate = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999

    D_real_scale = 1.0
    D_fake_scale = 0.0

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(img_size),
         transforms.Normalize([0.5], [0.5])])

    trainset = input_trainset

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                             shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            # Discriminator:
            self.conv1 = nn.Conv2d(3, 128, 4, 2, 1) # (bs, 3 + , img_size, img_size)
            self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
            self.conv2_bn = nn.BatchNorm2d(256)
            self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
            self.conv3_bn = nn.BatchNorm2d(512)
            self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)
            self.conv4_bn = nn.BatchNorm2d(1024)
            #self.conv5 = nn.Conv2d(1024, 1, 4, 1, 0)
            self.conv5 = nn.Conv2d(1024, 1, 2, 1, 0)
        
        
        def weight_init(self):
            for m in self._modules:
                normal_init(self._modules[m])
          
        def forward(self, x):
            x = F.leaky_relu(self.conv1(x), 0.2)
            x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
            x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
            x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
            x = torch.sigmoid(self.conv5(x))
            return x

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            ## Decoding:
            self.deconv1 = nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0, bias = False) # Not sure how this looks
            self.deconv1_bn = nn.BatchNorm2d(1024)
            self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False)
            self.deconv2_bn = nn.BatchNorm2d(512)
            self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False)
            self.deconv3_bn = nn.BatchNorm2d(256)
            self.deconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False)
            self.deconv4_bn = nn.BatchNorm2d(128)
            #self.deconv5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)
            self.deconv5 = nn.ConvTranspose2d(128, 3, 3, 1, 1)
        
        def weight_init(self):
            for m in self._modules:
                normal_init(self._modules[m])
          
        def forward(self, z):
            x = self.deconv1_bn(self.deconv1(z))
            x = F.relu(self.deconv2_bn(self.deconv2(x)))
            x = F.relu(self.deconv3_bn(self.deconv3(x)))
            x = F.relu(self.deconv4_bn(self.deconv4(x)))
            x = torch.tanh(self.deconv5(x))
            return x

    def normal_init(m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            #m.bias.data.zero_()

    def get_codes(size, hardware = device, hot = True):
        if hot == True:
            return one_hot_embedding(torch.randint(c_dim, size = (size, 1), device = hardware))
        
        else:
            return torch.randint(c_dim, size = (size, 1), device = hardware)

    def one_hot_embedding(labels):
        #y = torch.eye(num_classes)
        #return y[labels]
        #return torch.nn.functional.one_hot(labels)[:,1:]
        
        labels = torch.nn.functional.one_hot(torch.tensor(labels).to(torch.int64), num_classes = c_dim)
        return torch.squeeze(labels)

    def print_g_sample():
        with torch.no_grad():
            varis = torch.randn((9, z_dim,1,1), device = device) # walk from [0,...,0] to [1,...,1]
            #print(codes.shape, varis.shape)
            generated = .5*(G(varis).cpu() + 1)
            generated = torch.squeeze(generated)
            #print(generated.shape)
            for i in range(9):
                plt.subplot(330 + 1 + i)
                # plot raw pixel data
                element = generated[i,:].permute(1,2,0)
                plt.imshow(element, cmap = 'gray')
            plt.show()

    G = Generator()
    D = Discriminator()
    G.weight_init()
    D.weight_init()
    G.to(device)
    D.to(device)

    BCE_loss = nn.BCELoss()

    G_optimizer = optim.Adam(G.parameters(),
                             lr = learning_rate,
                             betas = (beta_1, beta_2))
                             
    D_optimizer = optim.Adam(D.parameters(),
                             lr = learning_rate,
                             betas = (beta_1, beta_2))

    G_loss_tracker, D_loss_tracker = [], []
    for epoch in range(1, n_epochs+1):

        D_losses = []
        G_losses = []
      
        for X, code in train_loader:
            #print(code)
            mini_batch = X.size()[0]

            X = X.to(device)
            
            ## Discriminator Training
            for param in D.parameters():
                param.grad = None
                
            
            y_real = torch.ones((mini_batch,1,1,1), device = device)*D_real_scale # Sometimes .9, .1
            y_fake = torch.ones((mini_batch,1,1,1), device = device)*D_fake_scale 
            
            rand_z = torch.randn((mini_batch, z_dim, 1, 1), device = device)
            
            #print(X.shape, code.shape)
            D_real_out = D(X)
            D_real_loss = BCE_loss(D_real_out, y_real)
            
            X_fake = G(rand_z)
            #print(X_fake.shape)
            D_fake_out = D(X_fake)
            D_fake_loss = BCE_loss(D_fake_out, y_fake)
            
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            D_optimizer.step()
            
            ## Generator Training
            for param in G.parameters():
                param.grad = None
                
            rand_z = torch.randn((mini_batch, z_dim, 1, 1), device = device)
            X_fake = G(rand_z)
            D_out = D(X_fake)
            y_targ = torch.ones((mini_batch,1,1,1), device = device) #G gets low loss when D returns X_fake near 1
            G_loss = BCE_loss(D_out, y_targ)

            ## Loss combination
            model_loss = G_loss
            model_loss.backward()
            G_optimizer.step()

            D_losses.append(D_loss.data.item())
            G_losses.append(G_loss.data.item())

        if epoch % print_stride == 0:
            print('Epoch {} - loss_D: {:.3f}, loss_G: {:.3f}'.format((epoch),
                                                                   torch.mean(torch.FloatTensor(D_losses)),
                                                                   torch.mean(torch.FloatTensor(G_losses))))

            G_loss_tracker.append(torch.mean(torch.FloatTensor(G_losses)))
            D_loss_tracker.append(torch.mean(torch.FloatTensor(D_losses)))
            if print_samp == True:
                print_g_sample()

    import torch
    torch.save(G.state_dict(), f'gan_cifar_quarter_{exp_num}_{n_epochs}e_G.pt')
    return D, G

