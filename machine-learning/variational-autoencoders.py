import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision

import numpy as np
import matplotlib.pyplot as plt


def imshow(img):

    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5

    img = img.numpy()
    img = np.transpose(img,(1, 2, 0))
    n_channels = img.shape[-1]

    if n_channels == 3 :
        plt.imshow(img)

    else :
        for channel in range(n_channels) :
            plt.figure()
            plt.imshow(img[:,:,channel])


class Net(nn.Module):

    def __init__(self, latent_dim=10):
        super(Net, self).__init__()

        self.latent_dim = latent_dim
        self.training = False

        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool1 = nn.MaxPool2d(2,return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool2 = nn.MaxPool2d(2,return_indices=True)
        # an affine operation: y = Wx + b

        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84) # fully connected bitsW
        self.fc3 = nn.Linear(84, 2*latent_dim) # variance and mean

        self.defc1 = nn.Linear(latent_dim,84) # input from encoder only mean+noise; no variance
        self.defc2 = nn.Linear(84,120) # fully connected bitsW
        self.defc3 = nn.Linear(120, 16 * 6 * 6)  # 6*6 from image dimension

        self.deconv1 = nn.ConvTranspose2d(16,6,3)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv2 = nn.ConvTranspose2d(6,3,3)
        self.unpool2 = nn.MaxUnpool2d(2)

    def encode(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x, self.pool_indices1 = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        print(x.shape)
        x, self.pool_indices2 = self.pool2(x)

        # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

    def decode(self, x):

        x = self.defc1(x)
        x = F.relu(x)
        x = self.defc2(x)
        x = F.relu(x)
        x = self.defc3(x)

        # reconstructing square number into 6x6 images, with 16xchannels
        x = x.view(-1,16,6,6)

        x = self.unpool1(x,self.pool_indices2,output_size=torch.Size([4,16,13,13]))
        x = F.relu(x)
        x = self.deconv1(x)

        x = self.unpool2(x,self.pool_indices1)
        x = F.relu(x)
        x = self.deconv2(x)

        return x

    def sample(self,x):

        mu = x[:,:self.latent_dim] # take all batches; first dim is batch size
        sigma = x[:,self.latent_dim:]

        if self.training :
            return torch.normal(mean=mu,std=sigma)
        else :
            return mu

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self,x):

        x = self.encode(x)
        x = self.sample(x)
        x = self.decode(x)

        return x

################################### load the data
transform = transforms.Compose( [transforms.ToTensor(),
    transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=8)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

images.min(),images.max()

# show sample images
imshow(images)

############### create the net
model = Net().to("cuda")
model.forward(images.to("cuda"))

# Setting the optimiser
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch

nn.functional.binary_cross_entropy( images, images, reduction='sum')

def loss_function(x_hat, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x.view(-1, 784), reduction='sum'
    )
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

    return BCE + KLD


# Training and testing the VAE

epochs = 10
codes = dict(μ=list(), logσ2=list(), y=list())
for epoch in range(0, epochs + 1):
    # Training
    if epoch > 0:  # test untrained net first
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            # ===================forward=====================
            x_hat, mu, logvar = model(x)
            loss = loss_function(x_hat, x, mu, logvar)
            train_loss += loss.item()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    
    # Testing
    
    means, logvars, labels = list(), list(), list()
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            # ===================forward=====================
            x_hat, mu, logvar = model(x)
            test_loss += loss_function(x_hat, x, mu, logvar).item()
            # =====================log=======================
            means.append(mu.detach())
            logvars.append(logvar.detach())
            labels.append(y.detach())
    # ===================log========================
    codes['μ'].append(torch.cat(means))
    codes['logσ2'].append(torch.cat(logvars))
    codes['y'].append(torch.cat(labels))
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    display_images(x, x_hat, 1, f'Epoch {epoch}')