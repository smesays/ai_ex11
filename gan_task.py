import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets, models, transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import os

# Hypeerparameters
BATCH_SIZE = 64
G_LR = D_LR = 1e-3
GLOBAL_STEP = 0
PRINT_EVERY = 1000
TOTAL_STEPS = 10000
COST_FUNC = nn.BCELoss().cuda()

D_NOISE = 100
D_SIDE = 28
D_IMG = D_SIDE**2
D_HIDDEN = 128

# Data Loader
modes = ['train', 'val']
trans = transforms.Compose([transforms.ToTensor(),]) # transforms.Normalize((0.1307,), (0.3081,))
dsets = {k: datasets.MNIST('./data', train=k=='train', download=True, transform=trans) for k in modes}
loaders = {k: torch.utils.data.DataLoader(dsets[k], batch_size=BATCH_SIZE, shuffle=True) for k in modes}

def random_noise(batch_size=64):
    return Variable(torch.randn(BATCH_SIZE, D_NOISE)).cuda()

def mnist():
    data = next(iter(loaders['train']))[0]
    return Variable(data).resize(BATCH_SIZE, D_IMG).cuda()

# Inspect Data
def imshow(inp, c, save=False, title=None):
    """Imshow for Tensor."""
    fig = plt.figure(figsize=(5, 5))
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    
    plt.title(title) if title is not None else plt.title(str(c).zfill(3))
    if save:
        if not os.path.exists('gan-out/'):
            os.makedirs('gan-out/')
        plt.savefig('gan-out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        plt.close(fig)

inputs = mnist().data.resize_(BATCH_SIZE, 1, D_SIDE, D_SIDE)
out = utils.make_grid(inputs)
imshow(out, c=0, save=False, title="Real MNIST digits")

# Build the discriminator (D) and generator (G) models
# TO DO

# Train
# TO DO

# Inspect Results
samples = G(random_noise(64)).data.resize_(64, 1, D_SIDE, D_SIDE)
samples = utils.make_grid(samples)
imshow(samples, c = GLOBAL_STEP // PRINT_EVERY, save=False,
       title="Fake MNIST digits ({} train steps)".format(GLOBAL_STEP))
