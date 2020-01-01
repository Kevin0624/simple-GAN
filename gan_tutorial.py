import sys
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils


import matplotlib.pyplot as plt


print("System version: ", end="")
print(sys.version)

print("torch version: ", end="")
print(torch.__version__)


''' Define show images function '''

def show_imgs(x, new_fig = True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0, 2).transpose(0, 1) # channels as last dimension

    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())
    plt.show()


''' Define the neural networks '''

''' Define a small 2-layers fully connected neural network for the Discriminator D '''
class Discriminator(nn.Module):

    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), 784) # flatten (batchsize x 1 x 28 x 28) -> (batchsize, 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out


''' Define a small 2-layers neural network for the generator G takes a 100-dimensional noise vector
    and generates an output of the size matching the data '''
class Generator(nn.Module):

    def __init__(self, z_dim = 100):
        super(Generator, self).__init__()
        self.fc1  = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)
    
    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out) # range [-1, 1]

        # convert to  image

        out = out.view(out.size(0), 1, 28, 28)
        return out



''' Show the weights of the discriminator and generator '''
D = Discriminator()
G = Generator()


print("Discriminator parameters : ")
for name, p in D.named_parameters():
    print(name, p.shape)


print("===============================")
print("Generator parameters : ")
for name, p in G.named_parameters():
    print(name, p.shape)


''' a small batch of 10 samples, random noise '''

z = torch.randn(10, 100)
x_gen = G(z)
print(x_gen.shape)
print(x_gen)
show_imgs(x_gen)


''' Lodaing the data and computing forward pass '''
dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))]), download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)


''' Get a batch of images and show them '''
batch, _ =iter(dataloader).next()
print(batch.shape)
show_imgs(batch)

''' Set optimizers and loss function '''
optimizer_D = torch.optim.SGD(D.parameters(), lr=0.01)
optimizer_G = torch.optim.SGD(D.parameters(), lr=0.01)

Loss_function = nn.BCELoss() # binary cross entropy


''' training loop '''
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)

''' initialize Discriminator and Generator in GPU or CPU '''
D = Discriminator().to(device)
G = Generator().to(device)

''' set up D, G  optimizer '''
optimizerD = torch.optim.Adam(D.parameters(), lr=0.0003)
optimizerG = torch.optim.Adam(G.parameters(), lr=0.0003)


lab_real = torch.ones(64, 1, device=device)
lab_fake = torch.zeros(64, 1, device=device)


''' for logging '''
collect_x_gen = []
fixed_noise = torch.randn(64, 100, device=device)
fig = plt.figure() # keep updating this one
plt.ion()

for epoch in range(2): ''' 10 epochs '''
    for i, data in enumerate(dataloader, 0):

        ''' STEP 1: Discriminator optimization step '''
        x_real, _ = iter(dataloader).next()
        x_real = x_real.to(device)

        '''reset accumulated gradients from previous iteration '''
        optimizerD.zero_grad()

        D_x = D(x_real)
        lossD_real = Loss_function(D_x, lab_real)

        z = torch.randn(64, 100, device=device) ''' random noise, 64 samples, z_dim=100 '''
        x_gen = G(z).detach()
        D_G_z = D(x_gen)
        lossD_fake = Loss_function(D_G_z, lab_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()
        
        ''' STEP 2: Generator optimization step '''
        ''' reset accumulated gradients from previous iteration '''
        optimizerG.zero_grad()

        z = torch.randn(64, 100, device=device) ''' random noise, 64 samples, z_dim=100 '''
        x_gen = G(z)
        D_G_z = D(x_gen)
        lossG = Loss_function(D_G_z, lab_real) 

        lossG.backward()
        optimizerG.step()
        if i % 100 == 0:
            x_gen = G(fixed_noise)
            
            print('e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(
                epoch, i, len(dataloader), D_x.mean().item(), D_G_z.mean().item()))
    # End of epoch
    x_gen = G(fixed_noise)
    collect_x_gen.append(x_gen.detach().clone())




