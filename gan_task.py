# Exercise 11 - Question 1
# by LIU, Jiun Ee
# modified from starter code and dcgan/main.py from PyTorch 
# I am using putty to access the CIP Pool to do this homework. I commented out all the imshow because it causes error.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import datasets, models, transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import os

# Hyperparameters
BATCH_SIZE = 64
G_LR = D_LR = 0.001
GLOBAL_STEP = 0
PRINT_EVERY = 10
TOTAL_STEPS = 20
#PRINT_EVERY = 1000
#TOTAL_STEPS = 10000
COST_FUNC = nn.BCELoss().cuda()

NUMBER_CHANNEL = 1
DIM_NOISE = 100
DIM_SIDE = 28
DIM_IMG = DIM_SIDE**2
DIM_HIDDEN = 128

# Data Loader
modes = ['train', 'val']
trans = transforms.Compose([transforms.ToTensor(),]) # transforms.Normalize((0.1307,), (0.3081,))
dsets = {k: datasets.MNIST('./data', train=k=='train', download=True, transform=trans) for k in modes}
loaders = {k: torch.utils.data.DataLoader(dsets[k], batch_size=BATCH_SIZE, shuffle=True) for k in modes}

def random_noise(batch_size=64):
    return Variable(torch.randn(BATCH_SIZE, DIM_NOISE)).cuda()

def mnist():
    data = next(iter(loaders['train']))[0]
    return Variable(data).resize(BATCH_SIZE, DIM_IMG).cuda()

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

inputs = mnist().data.resize_(BATCH_SIZE, 1, DIM_SIDE, DIM_SIDE)
out = utils.make_grid(inputs)
#imshow(out, c=0, save=False, title="Real MNIST digits")

# Build the discriminator (D) and generator (G) models

# custom weights initialization called on netG and netD
def weights_init(m):
    m.weight.data.normal_(0.0, 0.075**2)
    m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, input_shape=(DIM_SIDE, DIM_SIDE)):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(DIM_NOISE, 128),
            nn.ReLU(True),
            nn.Linear(128, np.prod(input_shape)*NUMBER_CHANNEL),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.generator(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_shape=(DIM_SIDE, DIM_SIDE)):
        super(Discriminator, self).__init__()        
        self.discriminator = nn.Sequential(
            nn.Linear(np.prod(input_shape)*NUMBER_CHANNEL, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.discriminator(input)
        return output.view(-1, 1).squeeze(1)


# Train
gen_model = Generator()
#gen_model.apply(weights_init) # this doesn't work. it is complaining ReLU does not have weight attribute. do not know how to fix
dis_model = Discriminator()
#dis_model.apply(weights_init)

input = torch.FloatTensor(BATCH_SIZE, NUMBER_CHANNEL, DIM_SIDE*DIM_SIDE)
noise = torch.FloatTensor(BATCH_SIZE, DIM_NOISE, 1, 1)
fixed_noise = torch.FloatTensor(BATCH_SIZE, DIM_NOISE, 1, 1).normal_(0, 1)
label = torch.FloatTensor(BATCH_SIZE)
real_label = 1
fake_label = 0

gen_model.cuda()
dis_model.cuda()
input.cuda()
noise.cuda()
fixed_noise.cuda()
label.cuda()

fixed_noise = Variable(fixed_noise)

gen_optim = optim.Adam(gen_model.parameters(), lr=G_LR)
dis_optim = optim.Adam(dis_model.parameters(), lr=D_LR)

for epoch in range(1, TOTAL_STEPS+1):
    for i, data in enumerate(loaders, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        dis_model.zero_grad()
	input = mnist().data.resize_(BATCH_SIZE, NUMBER_CHANNEL, DIM_SIDE*DIM_SIDE)
        label.resize_(BATCH_SIZE).fill_(real_label)
        input, label = Variable(input), Variable(label)

        output = dis_model(input)
	print output.size(), label.size() # this prints (64L), (64L)
        errD_real = COST_FUNC(output, label)
	# I get the following error on the above line:
	#   TypeError: is_same_size received an invalid combination of arguments - got (torch.cuda.FloatTensor), but expected (torch.FloatTensor other)
	#   confused and do not know how to continue to make this work... :(
	# I am not able to test any codes after this point due to the above error. 
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(BATCH_SIZE, Z_DIM, 1, 1).normal_(0, 1)
        noise = Variable(noise)
        fake = gen_model(noise)
        label = Variable(label.fill_(fake_label))
        output = dis_model(fake.detach())
        errD_fake = COST_FUNC(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        dis_optim.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        gen_model.zero_grad()
        label = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = dis_model(fake)
        errG = COST_FUNC(output, label)
        errG.backward()
        D_G_z2 = output.data.mean()
        gen_optim.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, TOTAL_STEPS, i, len(loaders),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % PRINT_EVERY == 0:
	    imshow(input.data,'Input',save=True)
            fake = gen_model(fixed_noise)
	    imshow(fake.data,'Fake',save=True)

    # do checkpointing
    torch.save(gen_model.state_dict(), 'gen_model_epoch_%d.pth' % epoch)
    torch.save(dis_model.state_dict(), 'dis_model_epoch_%d.pth' % epoch)
    
    # Generate 2000 fake images
    if epoch == TOTAL_STEPS: # At the last epoch, use the generative model to generate fake images?
        for zz in range(len(label)): # for each class
            # pick first 2 images
            img1 = input[label==zz][0]
#	    imshow(img1,'Image 1')
            img2 = input[label==zz][1]
#	    imshow(img2,'Image 2')

	    fake_img = np.empty(2000)
	    dist1 = np.empty(2000)
	    dist2 = np.empty(2000)
	    for zz in range(2000):
   	        fake_img[zz] = gen_model(fixed_noise)
	        # calculate euclidean distances
                dist1[zz] = np.linalg.norm(img1 - fake_img[zz])
	        dist2[zz] = np.linalg.norm(img2 - fake_img[zz])

	    dist1_rank = np.argsort(dist1)
	    dist2_rank = np.argsort(dist2)
	    for zz in range(4): # get the 4 nearest images
		fake=fake_img[dist1_rank[zz]]
#		imshow(fake, 'Fake Image')

# Inspect Results
samples = G(random_noise(64)).data.resize_(64, 1, DIM_SIDE, DIM_SIDE)
samples = utils.make_grid(samples)
#imshow(samples, c = GLOBAL_STEP // PRINT_EVERY, save=False,
#       title="Fake MNIST digits ({} train steps)".format(GLOBAL_STEP))



