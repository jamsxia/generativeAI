## James Xia GANs, finally on GPU
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import os
import math
import torch.nn.functional as F
##pulling data:Download and load the Fashion-MNIST dataset 
# using `torchvision.datasets.FashionMNIST`.
torch.manual_seed(42)
##Normalize the images to have values between -1 and 1, as this range typically works well with
##the ReLU activation function used in the generator's layers.
# Arguments

BATCH_SIZE = 256
EPOCHS = 500
Z_DIM = 10
LOAD_MODEL = False
CHANNELS = 1
DB = 'FashionMNIST'
IMAGE_SIZE = 28

def generate_imgs(z, epoch=0):
    generator.eval()
    fake_imgs = generator(z).detach().cpu()
    fake_imgs_ = vutils.make_grid(fake_imgs, normalize=True, nrow=math.ceil(BATCH_SIZE ** 0.5))
    plt.axis("off")
    plt.imshow(fake_imgs.numpy().transpose((1,2,0)),cmap="gray_r")
    plt.axis("off")
    plt.show()
# Data loaders
'''transform=T.Compose([
    #T.Resize(image_size),
    T.ToTensor(),
    T.Normalize([0.5],[0.5])
])'''
mean = np.array([0.5])
std = np.array([0.5])
transform = transforms.Compose([transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
train_set=torchvision.datasets.FashionMNIST(
    root=".",
    train=True,
    download=True,
    transform=transform
)
## Create DataLoader objects for the training dataset to iterate over batches.

train_loader=torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# Fix images for viz
fixed_z = torch.randn(BATCH_SIZE, Z_DIM).cuda()

# Labels
real_label = torch.ones(BATCH_SIZE).reshape([BATCH_SIZE,1]).cuda()
fake_label = torch.zeros(BATCH_SIZE).reshape([BATCH_SIZE,1]).cuda()
total_iters = 0
max_iter = len(train_loader)
# Networks
class Generator(nn.Module):
    def __init__(self, z_dim=10, image_size=28, channels=1, h_dim=1024):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, image_size * image_size * channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x.reshape([-1, self.channels, self.image_size, self.image_size])
        return x


class Discriminator(nn.Module):
    def __init__(self, image_size=28, channels=1, h_dim=256):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(image_size * image_size * channels, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()

    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
                nn.Linear(784, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
        )
        
    def forward(self, x):
        # x is batch_size*28*28 dim tensor if sampled from the Fashion-MNIST
        # if its from the generator, a 784 tensor
        x = x.view(x.size(0), 784)
        out = self.model(x)
        return out
    
generator = Generator().cuda()
discriminator = Discriminator().cuda()

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
img_list = []
for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()

    for i, data in enumerate(train_loader):

        total_iters += 1

        # Loading data
        x_real, _ = data
        z_fake = torch.randn(BATCH_SIZE, Z_DIM)
        x_real = x_real.cuda()
        z_fake = z_fake.cuda()

        # generatorerate fake data
        x_fake = generator(z_fake)

        # Train Discriminator
        fake_out = discriminator(x_fake.detach())
        real_out = discriminator(x_real.detach())
        #print(fake_label,fake_out)
        d_loss = (criterion(fake_out, fake_label) + criterion(real_out, real_label)) / 2

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train generatorerator
        fake_out = discriminator(x_fake)
        g_loss = criterion(fake_out, real_label)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if i % 50 == 0:
            print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
                  + "\titer: " + str(i) + "/" + str(max_iter)
                  + "\ttotal_iters: " + str(total_iters)
                  + "\td_loss:" + str(round(d_loss.item(), 4))
                  + "\tg_loss:" + str(round(g_loss.item(), 4))
                  )
            with torch.no_grad():
                fake = generator(z_fake).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))


        #generate_imgs(fixed_z, epoch=epoch + 1)

#generate_imgs(fixed_z)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

