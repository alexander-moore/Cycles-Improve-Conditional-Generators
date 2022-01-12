## Generative models to load
import torch
import torch.nn as nn
import torch.nn.functional as F

class rgb_32_C_G(nn.Module):
    def __init__(self, v_dim, c_dim):
        super(rgb_32_C_G, self).__init__()
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
        self.deconv5 = nn.ConvTranspose2d(128, 3, 1, 1, 0)
    
    
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
      
    def forward(self, v, c):
        v = self.deconv1_bn(self.deconv1v(v))
        c = self.deconv1_bn(self.deconv1c(c))
        x = torch.cat((v, c), dim = 1) #stack on channel dim, should be (bs, vdim+cdim, 1, 1). Not sure here
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        return x

class rgb_32_G(nn.Module):
    def __init__(self, z_dim):
        super(rgb_32_G, self).__init__()
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

class rgb_32_C_AE(nn.Module):
    # CAEGAN for 32x32 rgb (cifar10)
    def __init__(self, v_dim, c_dim):
        super(rgb_32_C_AE, self).__init__()
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
        c = torch.sigmoid(self.conv5c(x)) # this is softmax for CLASSIFICATION. Shapes3d is not 1-classif..
        
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

class rgb_32_C_AE_smax(nn.Module):
    # CAEGAN for 32x32 rgb (cifar10)
    def __init__(self, v_dim, c_dim):
        super(rgb_32_C_AE_smax, self).__init__()
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

class rgb_32_C_IAE(nn.Module):
    def __init__(self, v_dim, c_dim):
        super(rgb_32_C_IAE, self).__init__()
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
    
    def pass_thru(self, v, c):
        x = self.forward(v, c)
        return self.encode(x)

def normal_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)

class rgb_32_AE(nn.Module):

    def __init__(self, z_dim):
        super(rgb_32_AE, self).__init__()
        
        ## Encoding
        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1, bias = False)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1, bias = False)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, z_dim, 2, 1, 0)
        
        ## Decoding
        self.deconv1 = nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0, bias = False)
        self.deconv1_bn = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False)
        self.deconv2_bn = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False)
        self.deconv3_bn = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False)
        self.deconv4_bn = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 3, 3, 1, 1)

        learning_rate = 0.0002
        beta_1 = 0.5
        beta_2 = 0.99

        z_dim = 100

        batch_size = 8
        n_epochs = 10

        n_updates = 62500



    
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
            
    def encode(self, x):
        # Encode data x to 2 spaces: condition space and variance-space
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        z = self.conv5(x) # (-1,1) tanh, times 3 gives (-3, 3), or 99% of randn variability
        
        return z
      
    def forward(self, z): #forward is decoding for generalizability
        x = self.deconv1_bn(self.deconv1(z))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x)) #changed from -1,1 to 0,1 somehow?
        return x
    
    def pass_thru(self, x):
        z = self.encode(x)
        return self.forward(z)

class rgb_32_D(nn.Module):
    def __init__(self):
        super(rgb_32_D, self).__init__()
        # Discriminator:
        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1) # (bs, 3 + , img_size, img_size)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, 2, 1, 0)
        # possible i need to finagle some kind of conv6... not sure
    
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
      
    def forward(self, x):
        #print(x.shape, c.shape)
        #print(x.shape)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        
        return x

class rgb_32_C_D(nn.Module):
    def __init__(self, v_dim, c_dim, img_size = 32):
        super(rgb_32_C_D, self).__init__()
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

class rgb_32_AC_D(nn.Module):
    def __init__(self, c_dim):
        super(rgb_32_AC_D, self).__init__()
        
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

class rgb_32_small_classifier(nn.Module):
    # CAEGAN for 32x32 rgb (cifar10)
    def __init__(self, c_dim):
        super(rgb_32_small_classifier, self).__init__()
        ## Encoding: Unconditional samples
        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1) # Input: (bs, 3, img_size, img_size)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
        self.conv2_bn = nn.BatchNorm2d(256)
        
        self.fc= nn.Linear(256*8*8, c_dim) # Output: (bs, c_dim, 1, 1)
        

    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
            
    def forward(self, x):
        # Encode data x to 2 spaces: condition space and variance-space
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = x.view(x.shape[0], -1)
        #print(x.shape, self.fc.weight.shape)
        v = nn.Softmax(dim=1)(self.fc(x)) # Variance-space unif~[0,1]        
        return v
      
class rgb_32_big_classifier(nn.Module):
    def __init__(self, c_dim):
        super(rgb_32_big_classifier, self).__init__()
        ## Encoding: Unconditional samples
        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1) # Input: (bs, 3, img_size, img_size)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1, bias = False)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1, bias = False)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.conv5c = nn.Conv2d(1024, c_dim, 2, 1, 0) # Output, same as above: but this one to condition-space
        
    
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
            
    def forward(self, x):
        # Encode data x to 2 spaces: condition space and variance-space
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        
        c = torch.nn.Softmax(dim=1)(self.conv5c(x)) # this is softmax for CLASSIFICATION. Shapes3d is not 1-classif..
        
        return c
      