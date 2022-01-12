def train_cyc_caegan(exp_num, n_epochs, input_trainset, print_samp = True):
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
    n_epochs = 10 #50? depends on max_sampels
    print_stride = 10
    bs = 16 # 64

    v_dim = 100
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
            # Condition Up-Embedder:
            self.fc1 = nn.Linear(c_dim, img_size**2, bias = False)
            # Discriminator:
            self.conv1 = nn.Conv2d(4, 128, 4, 2, 1) # (bs, 3 + , img_size, img_size)
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
          
        def forward(self, x, c):
            c = torch.tanh(self.fc1(c.view(mini_batch, c_dim))).view(mini_batch, 1, img_size, img_size) # Tanh: Since x is in (-1,1), c should probably too
            #print(x.shape, c.shape)
            
            x = torch.cat((x, c), dim = 1)
            x = F.leaky_relu(self.conv1(x), 0.2)
            x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
            x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
            x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
            x = torch.sigmoid(self.conv5(x))
            return x

    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()
            ## Encoding: Unconditional samples
            self.conv1 = nn.Conv2d(3, 128, 4, 2, 1) # Input: (bs, 3, img_size, img_size)
            self.conv2 = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
            self.conv2_bn = nn.BatchNorm2d(256)
            self.conv3 = nn.Conv2d(256, 512, 4, 2, 1, bias = False)
            self.conv3_bn = nn.BatchNorm2d(512)
            self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1, bias = False)
            self.conv4_bn = nn.BatchNorm2d(1024)
            
            self.conv5v = nn.Conv2d(1024, v_dim, 2, 1, 0) # Output: (bs, c_dim, 1, 1)
            self.conv5c = nn.Conv2d(1024, c_dim, 2, 1, 0) # Output, same as above: but this one to condition-space
            
            ## Decoding:
            self.deconv1v = nn.ConvTranspose2d(v_dim, 1024, 4, 1, 0, bias = False) # Not sure how this looks
            self.deconv1c = nn.ConvTranspose2d(c_dim, 1024, 4, 1, 0, bias = False) # Input: (bs, cdim+v_dim, 1, 1)
            
            self.deconv1_bn = nn.BatchNorm2d(1024)
            self.deconv2 = nn.ConvTranspose2d(1024+1024, 512, 4, 2, 1, bias = False)
            self.deconv2_bn = nn.BatchNorm2d(512)
            self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False)
            self.deconv3_bn = nn.BatchNorm2d(256)
            self.deconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False)
            self.deconv4_bn = nn.BatchNorm2d(128)
            self.deconv5 = nn.ConvTranspose2d(128, 3, 3, 1, 1)
        
        def weight_init(self):
            for m in self._modules:
                normal_init(self._modules[m])
                
        def encode(self, x):
            # Encode data x to 2 spaces: condition space and variance-space
            x = F.leaky_relu(self.conv1(x), 0.2)
            x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
            x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
            x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
            
            v = torch.sigmoid(self.conv5v(x)) # Variance-space unif~[0,1]
            c = torch.nn.Softmax(dim=1)(self.conv5c(x)) # this is softmax for CLASSIFICATION. Shapes3d is not 1-classif..
            
            return v, c
          
        def forward(self, v, c):
            # This is actually conditional generation // decoding.
            # It's beneficial to call this forward, though, for FJD calculation
            v = self.deconv1_bn(self.deconv1v(v))
            c = self.deconv1_bn(self.deconv1c(c))
            x = torch.cat((v, c), dim = 1) #stack on channel dim, should be (bs, vdim+cdim, 1, 1). Not sure here
            x = F.relu(self.deconv2_bn(self.deconv2(x)))
            x = F.relu(self.deconv3_bn(self.deconv3(x)))
            x = F.relu(self.deconv4_bn(self.deconv4(x)))
            x = torch.tanh(self.deconv5(x))
            return x
        
        def pass_thru(self, x):
            v, c = self.encode(x)
            return self.forward(v, c)

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
            codes = one_hot_embedding(torch.tensor(list(range(9)), device = device)).view(9,c_dim,1,1).float()
            varis = torch.randn((9, v_dim,1,1), device = device) # walk from [0,...,0] to [1,...,1]
            #print(codes.shape, varis.shape)
            generated = .5*(AE.forward(varis, codes).cpu() + 1)
            generated = torch.squeeze(generated)
            #print(generated.shape)
            for i in range(9):
                plt.subplot(330 + 1 + i).set_title(str(classes[i]))
                # plot raw pixel data
                element = generated[i,:].permute(1,2,0)
                plt.imshow(element, cmap = 'gray')
            plt.show()

    def test_acc():
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                #print('True labels', labels)
                _, outputs = AE.encode(images.to(device))
                #print(outputs)
                _, predicted = torch.max(outputs.data, 1)
                #print('Predicted Labels', predicted.T)
                
                total += images.shape[0]
                #print('Total smaples', total)
                correct += (predicted.T == labels.to(device)).sum().item()
                #print((predicted == labels.to(device)).shape)
                #print('Correct preds', correct)
                

        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
        
        return 100 * correct / total

    AE = Autoencoder()
    D = Discriminator()
    AE.weight_init()
    D.weight_init()
    AE.to(device)
    D.to(device)

    BCE_loss = nn.BCELoss()
    MSE_loss = nn.MSELoss()

    AE_optimizer = optim.Adam(AE.parameters(),
                             lr = learning_rate,
                             betas = (beta_1, beta_2))
                             
    D_optimizer = optim.Adam(D.parameters(),
                             lr = learning_rate,
                             betas = (beta_1, beta_2))

    G_loss_tracker, D_loss_tracker, AE_loss_tracker = [], [], []
    c_loss_tracker = []
    for epoch in range(1, n_epochs+1):

        D_losses = []
        G_losses = []
        recon_losses, c_losses = [], []
      
        for X, code in train_loader:
            mini_batch = X.size()[0]
            #print(X.shape, torch.min(X), torch.max(X))
            #print(code)
            #break
            #X = torch.squeeze(X).to(device)
            #X = X.unsqueeze(1).to(device)
            X = X.to(device)
            #print(X.shape)
            #break
            code = code.to(device)
            code = one_hot_embedding(code).float()

            
            ## Discriminator Training
            for param in D.parameters():
                param.grad = None
                
            
            y_real = torch.ones((mini_batch,1,1,1), device = device)*D_real_scale # Sometimes .9, .1
            y_fake = torch.ones((mini_batch,1,1,1), device = device)*D_fake_scale 
            
            rand_v = torch.randn((mini_batch, v_dim, 1, 1), device = device)
            rand_c = get_codes(mini_batch).view(mini_batch, c_dim, 1, 1).float()
            
            #print(X.shape, code.shape)
            D_real_out = D(X, code)
            D_real_loss = BCE_loss(D_real_out, y_real)
            
            X_fake = AE.forward(rand_v, rand_c)
            D_fake_out = D(X_fake, rand_c)
            D_fake_loss = BCE_loss(D_fake_out, y_fake)
            
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            D_optimizer.step()
            
            ## Generator Training

            for param in AE.parameters():
                param.grad = None
            


            rand_v = torch.randn((mini_batch, v_dim, 1, 1), device = device)
            rand_c = get_codes(mini_batch).view(mini_batch, c_dim, 1, 1).float()
            X_fake = AE.forward(rand_v, rand_c)
            D_out = D(X_fake, rand_c)
            y_targ = torch.ones((mini_batch,1,1,1), device = device) #G gets low loss when D returns X_fake near 1
            G_loss = BCE_loss(D_out, y_targ)

            ## Cycle Loss
            v_hat, c_hat = AE.encode(X_fake)
            c_loss = BCE_loss(c_hat, rand_c)
            v_cyc_loss = MSE_loss(v_hat, rand_v)

            
            ## Autoencoder Training
            X_hat = AE.pass_thru(X)
            recon_loss = MSE_loss(X_hat, X)
            
            ## Latent-Structure Loss
            v, c = AE.encode(X)
            condition_loss = c_loss + BCE_loss(c.view(mini_batch, c_dim), code)



            ## Loss combination
            model_loss = G_loss + recon_loss + condition_loss + v_cyc_loss #condition is c-cycle and c-encode. maybe dont even need condition loss then..
            model_loss.backward()
            AE_optimizer.step()

            D_losses.append(D_loss.data.item())
            G_losses.append(G_loss.data.item())
            recon_losses.append(recon_loss.data.item())
            c_losses.append(condition_loss.data.item())
        if print_samp == True:
            if epoch % print_stride == 0:
                print('Epoch {} - loss_ae (mse): {:.3f}, loss_cond: {:.3f}, loss_D: {:.3f}, loss_G: {:.3f}'.format((epoch),
                                                                       torch.mean(torch.FloatTensor(recon_losses)),
                                                                       torch.mean(torch.FloatTensor(c_losses)),
                                                                       torch.mean(torch.FloatTensor(D_losses)),
                                                                       torch.mean(torch.FloatTensor(G_losses))))

                AE_loss_tracker.append(torch.mean(torch.FloatTensor(recon_losses)))
                c_loss_tracker.append(torch.mean(torch.FloatTensor(c_losses)))
                G_loss_tracker.append(torch.mean(torch.FloatTensor(G_losses)))
                D_loss_tracker.append(torch.mean(torch.FloatTensor(D_losses)))
                
                print_g_sample()

    acc = test_acc()
    torch.save(AE.state_dict(), f'cyc_caegan_cifar_{n_epochs}_{exp_num}_G.pt')
    return D, AE, acc

