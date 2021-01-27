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

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


################################### load the data
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=8)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show sample images
imshow(images)

f = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=[5,5]) # defines 2d conv 1dim input 6dims

images.shape
f(images).shape

imshow(f(images).detach())

model = torch.hub.load('pytorch/vision:v0.8.2', 'alexnet', pretrained=True)

model.features[0]
model.features[0].weight.shape

imshow(model.features[0].weight.detach())

#imshow(model.features[3].weight.detach())

############### create the net
net = Net()
print(net)