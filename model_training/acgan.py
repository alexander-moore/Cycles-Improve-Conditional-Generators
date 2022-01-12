def train_acgan(exp_num, n_epochs, input_trainset, print_samp = True):
	#!/usr/bin/env python
	# coding: utf-8

	# In[1]:


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

	import model_to_fid
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)


	# In[2]:



	## All Hyperparams should go here. CGAN, CAEGAN, ICAEGAN
	img_size = 32 # can use this to mofidy data size to fit this model
	n_epochs = n_epochs #50? depends on max_sampels
	print_stride = 1
	bs = 16 # 64

	v_dim = 100
	c_dim = 10

	learning_rate = 0.0002
	beta_1 = 0.5
	beta_2 = 0.999

	D_real_scale = 1.0
	D_fake_scale = 0.0


	# In[3]:


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


	# In[4]:


	class AC_Discriminator(nn.Module):
	    def __init__(self):
	        super(AC_Discriminator, self).__init__()
	        
	        # Discriminator:
	        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1) # (bs, 3 + , img_size, img_size)
	        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
	        self.conv2_bn = nn.BatchNorm2d(256)
	        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
	        self.conv3_bn = nn.BatchNorm2d(512)
	        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)
	        self.conv4_bn = nn.BatchNorm2d(1024)
	        self.conv5 = nn.Conv2d(1024, 1, 4, 1, 0)
	        
	        self.fc_adv = nn.Linear(4096, 1)
	        self.fc_aux = nn.Linear(4096, c_dim)
	    
	    def weight_init(self):
	        for m in self._modules:
	            normal_init(self._modules[m])
	      
	    def forward(self, x):
	        x = F.leaky_relu(self.conv1(x), 0.2)
	        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
	        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
	        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
	        #print('after conv4:', x.shape)
	        x = nn.Flatten()(x)
	        #x = torch.sigmoid(self.conv5(x))
	        validity = torch.sigmoid(self.fc_adv(x))
	        label = nn.Softmax(dim=1)(self.fc_aux(x))
	        return validity, label

	class Generator(nn.Module):
	    def __init__(self):
	        super(Generator, self).__init__()
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
	        
	        #self.deconv5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)
	        self.deconv5 = nn.ConvTranspose2d(128, 3, 1, 1, 0)
	    
	    
	    def weight_init(self):
	        for m in self._modules:
	            normal_init(self._modules[m])
	      
	    def forward(self, v, c):
	        v = self.deconv1_bn(self.deconv1v(v))
	        c = self.deconv1_bn(self.deconv1c(c))
	        x = torch.cat((v, c), dim = 1) #stack on channel dim, should be (bs, vdim+cdim, 1, 1). Not sure here
	        x = F.relu(self.deconv2_bn(self.deconv2(x)))
	        #print('after2', x.shape)
	        x = F.relu(self.deconv3_bn(self.deconv3(x)))
	        #print('after3', x.shape)
	        x = F.relu(self.deconv4_bn(self.deconv4(x)))
	        #print('after4', x.shape)
	        x = torch.tanh(self.deconv5(x))
	        #print('after5', x.shape)
	        return x


	# In[5]:


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


	# In[6]:


	def print_g_sample():
	    with torch.no_grad():
	        codes = one_hot_embedding(torch.tensor(list(range(9)), device = device)).view(9,c_dim,1,1).float()
	        varis = torch.randn((9, v_dim,1,1), device = device) # walk from [0,...,0] to [1,...,1]
	        #print(codes.shape, varis.shape)
	        generated = .5*(G(varis, codes).cpu() + 1)
	        #generated = torch.squeeze(generated)
	        #print(generated.shape)
	        for i in range(9):
	            plt.subplot(330 + 1 + i).set_title(str(classes[i]))
	            # plot raw pixel data
	            element = generated[i,:].permute(1,2,0)
	            plt.imshow(element, cmap = 'gray')
	        plt.show()


	# In[7]:




	G = Generator()
	D = AC_Discriminator()
	G.weight_init()
	D.weight_init()
	G.to(device)
	D.to(device)

	adversarial_loss = nn.BCELoss()
	auxiliary_loss = nn.BCELoss()

	G_optimizer = optim.Adam(G.parameters(),
	                         lr = learning_rate,
	                         betas = (beta_1, beta_2))
	                         
	D_optimizer = optim.Adam(D.parameters(),
	                         lr = learning_rate,
	                         betas = (beta_1, beta_2))


	# In[ ]:


	G_loss_tracker, D_loss_tracker = [], []
	for epoch in range(1, n_epochs+1):

	    D_losses = []
	    G_losses = []
	  
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
	        
	        valid = Variable(torch.FloatTensor(mini_batch, 1).fill_(1.0), requires_grad=False).to(device)
	        fake = Variable(torch.FloatTensor(mini_batch, 1).fill_(0.0), requires_grad=False).to(device)
	        
	        ## Discriminator Training
	        for param in D.parameters():
	            param.grad = None


	            
	        real_pred, real_aux = D(X)
	        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, code)) / 2
	        
	        rand_v = torch.randn((mini_batch, v_dim, 1, 1), device = device)
	        rand_c = get_codes(mini_batch).view(mini_batch, c_dim, 1, 1).float()
	        
	        X_fake = G(rand_v, rand_c)

	        fake_pred, fake_aux = D(X_fake)
	        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, rand_c.squeeze())) / 2
	        
	        d_loss = (d_real_loss + d_fake_loss) / 2

	        D_loss = d_loss
	        D_loss.backward()
	        D_optimizer.step()

	        ## Generator Training
	        for param in G.parameters():
	            param.grad = None
	            
	        rand_v = torch.randn((mini_batch, v_dim, 1, 1), device = device)
	        rand_c = get_codes(mini_batch).view(mini_batch, c_dim, 1, 1).float()
	        
	        X_fake = G(rand_v, rand_c)
	        
	        validity, pred_label = D(X_fake)
	        G_loss = 0.5*(adversarial_loss(validity, valid) + auxiliary_loss(pred_label, rand_c.squeeze()))
	        
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
	        #model_to_fid.get_fid_fjd(G, samples_per_condition = 1)


	#

	# In[ ]:

	torch.save(G.state_dict(), f'acgan_cifar_quarter_{exp_num}_{n_epochs}e_edited_G.pt')
	return D, G




