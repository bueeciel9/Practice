# Variables Autoencoder

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# 1. Load Data

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)

# 2. Define Model


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()

# 3. Define Loss and Optimizer

reconstruction_function = nn.BCELoss(size_average=False)
reconstruction_function.size_average = False


def loss_function(recon_x, x, mu, logvar):

    BCE = reconstruction_function(recon_x, x.view(-1, 784))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    # Normalise by same number of elements as in reconstruction
    # BCE /= x.view(-1, 784).size(0)
    # KLD /= x.view(-1, 784).size(0)

    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Train Model


def train(epoch):

    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):

        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return train_loss / len(train_loader.dataset)

# 5. Test Model


def test(epoch):

    model.eval()
    test_loss = 0

    for i, (data, _) in enumerate(test_loader):

        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(100, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss

# 6. Train and Test Model


train_loss = []
test_loss = []

for epoch in range(1, 51):
    train_loss.append(train(epoch))
    test_loss.append(test(epoch))

    sample = Variable(torch.randn(64, 20))
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')

# 7. Plot Loss

plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.legend()
plt.show()

# 8. Save Model

torch.save(model.state_dict(), 'VAE.pth')

# 9. Load Model

model.load_state_dict(torch.load('VAE.pth'))

# 10. Generate Images

sample = Variable(torch.randn(64, 20))
sample = model.decode(sample).cpu()
save_image(sample.data.view(64, 1, 28, 28),
           'results/sample_' + str(epoch) + '.png')

# 11. Plot Images
