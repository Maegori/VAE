import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
import random

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tf = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=tf
)

test_dataset = datasets.MNIST(
    './data',
    train=False,
    download=True,
    transform=tf
)

BATCH_SIZE = 64     # number of data points in each batch
N_EPOCHS = 10       # times to run the model on complete data
INPUT_DIM = 28 * 28 # size of each input
HIDDEN_DIM = 256    # hidden dimension
LATENT_DIM = 20     # latent vector dimension
lr = 1e-3           # learning rate

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim, z_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var

class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, z_dim, hidden_dim, output_dim):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
        '''
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]

        predicted = torch.sigmoid(self.out(hidden))
        # predicted is of shape [batch_size, output_dim]

        return predicted

class VAE(nn.Module):
    ''' 
    This the VAE, which takes an encoder and decoder.
    '''
    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        predicted = self.dec(x_sample)
        return predicted, z_mu, z_var

encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
model = VAE(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

def train():
    model.train()

    train_loss = 0
    for _, (x, _) in enumerate(train_iterator):
            # reshape the data into [batch_size, 784]
            x = x.view(-1, 28 * 28)
            x = x.to(device)
            
            optimizer.zero_grad()

            x_sample, z_mu, z_var = model(x)

            recon_loss = F.binary_cross_entropy(x_sample, x, size_average=False)
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
            loss = recon_loss + kl_loss

            loss.backward()
            train_loss += loss.item()
            
            optimizer.step()

    return train_loss

def test():
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for _, (x, _) in enumerate(test_iterator):
            x = x.view(-1, 28 * 28)
            x = x.to(device)

            x_sample, z_mu, z_var = model(x)

            recon_loss = F.binary_cross_entropy(x_sample, x, size_average=False)
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
            
            loss = recon_loss + kl_loss
            test_loss += loss.item()

    return test_loss


def trainer():

    best_test_loss = float('inf')

    for e in range(N_EPOCHS):

        train_loss = train()
        test_loss = test()

        train_loss /= len(train_dataset)
        test_loss /= len(test_dataset)

        print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

        if best_test_loss > test_loss:
            best_test_loss = test_loss
            patience_counter = 1
        else:
            patience_counter += 1

        if patience_counter > 3:
            break

    torch.save(model, "model")
def tester():
    device = torch.device('cpu')
    model = torch.load("model").to(device)

    fig, ax = plt.subplots()
    ax.margins(x=0)
    ax.imshow( model.dec(torch.zeros(LATENT_DIM)).view(28, 28).data, cmap='gray')
    
    slider_axes = [plt.axes([0.8, 0.1 + 0.04 * x, 0.15, 0.03], facecolor='lightgoldenrodyellow') for x in range(LATENT_DIM)]
    sliders = [Slider(slider_axes[x], str(20 - x), -10.0, 10.0, valinit=0, valstep=0.1) for x in range(LATENT_DIM)]

    def update(val):
        ax.clear()
        ax.imshow(model.dec(torch.Tensor([S.val for S in sliders])).view(28, 28).data, cmap='gray')
        fig.canvas.draw_idle()
        
    for S in sliders:
        S.on_changed(update)
        S.drawon = False

    ranax = plt.axes([0.05, 0.05, 0.2, 0.03])
    buttonRan = Button(ranax, "Random", color='lightgoldenrodyellow', hovercolor='0.975')

    def ran(event):
        for S in sliders:
            S.set_val(random.uniform(-10, 10))
    buttonRan.on_clicked(ran)

    resax = plt.axes([0.05, 0.1, 0.2, 0.03])
    buttonRes = Button(resax, "Reset", color='lightgoldenrodyellow', hovercolor='0.975')

    def reset(event):
        for S in sliders:
            S.reset()
    buttonRes.on_clicked(reset)
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python numgen.py <train/test>")
        sys.exit()

    if sys.argv[1] == "train":
        trainer()
    elif sys.argv[1] == "test":
        tester()