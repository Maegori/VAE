import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import sys
import random
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import numpy as np

from util.Trainer import Trainer
from midi import arry2mid

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()
torch.set_printoptions(threshold=5000)

MIDI_MAT = 88 * 96 * 16
N_MEASURES = 16
N_EPOCHS = 2
BATCH_SIZE = 350
SEED = 42
LR = 1e-3
EPS = 1e-8

ROOT_PATH = "data/midi/"
LOG_PATH = "/logs/VAE_musicgen_log"
MOD_PATH = "/models/VAE_musicgen_model"

class MidiDataset(Dataset):

    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        folder = os.listdir(self.root_dir)

        with open(self.root_dir + folder[idx], "rb") as tp:
            X = torch.load(tp)
        return X.reshape(-1)

class Encoder1(nn.Module):
    """
    Pre-encoder of the network, one is created for each measure.
    """

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(MIDI_MAT, 2000),
            nn.ReLU(),
            nn.Linear(2000, N_MEASURES * 200),
            nn.ReLU()
        )

    def forward(self, x):

        return self.encoder(x)

class Encoder2(nn.Module):
    """
    Second encoder of the network, all the pre-encoders are brought together into a 
    single latent space here.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(N_MEASURES * 200, 1600), 
            nn.ReLU(),
        )

        self.mu = nn.Linear(1600, 120)
        self.sigma = nn.Linear(1600, 120)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        sigma = self.sigma(x)

        return mu, sigma

class Decoder1(nn.Module):
    """
    First decoder, we inflate the main latent space.
    """

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(120, 1600),
            nn.ReLU(),
            nn.Linear(1600, N_MEASURES * 200),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(x)

class Decoder2(nn.Module):
    """
    Second decoder, where we inflate the data back to our MIDI measures.
    """

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(N_MEASURES * 200, 2000),
            nn.ReLU(),
            nn.Linear(2000, MIDI_MAT),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(x)

class VAE(nn.Module):
    """
    In this class we bring the four encoders and decoders together with reparameterisation.
    """

    def __init__(self, encoder1, encoder2, decoder1, decoder2):
        super().__init__()

        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder1 = decoder1
        self.decoder2 = decoder2
    
    def forward(self, x):
        mu, sigma = self.encoder2(self.encoder1(x))

        std = torch.exp(sigma / 2)
        eps = torch.randn_like(std)
        sample = eps.mul(std).add_(mu)
        
        y = self.decoder2(self.decoder1(sample))
        return y, mu, sigma

    def decoder(self, sample):
        return self.decoder2(self.decoder1(sample))
    
class VAETrainer(Trainer):
    def __init__(self, model, optimizer, criterion, trainloader, testloader, logPath):
        super().__init__(model, optimizer, criterion, trainloader, testloader, logPath)

    def calc_loss(self, *args):
        x, = args
        x = x.view(-1, MIDI_MAT)
        y, mu, sigma = self.model(x)
  
        try:
            reconLoss = self.crit(y+EPS, x)
        except:
            print(y)

        KLLoss = 0.5 * \
            torch.sum(torch.exp(sigma) + mu**2 - 1.0 - sigma)
        return reconLoss + KLLoss

def tester():
    tf = transforms.ToTensor()
    data = MidiDataset("data/midi/", tf)

    train_set, test_set = torch.utils.data.random_split(data, [105, 25])

    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)

    model = VAE(Encoder1(), Encoder2(), Decoder1(), Decoder2())

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss(reduction='sum')
    trainer = VAETrainer(model, optimizer, criterion, train_loader, test_loader, LOG_PATH)

    trainer._load_checkpoint("model")
    output = trainer.model.decoder(torch.randn((120)))

    threshold = nn.Threshold(0.01, 1.0)
    arry2mid(threshold(output.detach().reshape((96*16, 88))), "epoch1.mid")


if __name__ == "__main__":

    if sys.argv[1] == 'train':
        tf = transforms.ToTensor()
        data = MidiDataset("data/midi/", tf)

        train_set, test_set = torch.utils.data.random_split(data, [105, 25])

        train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)

        model = VAE(Encoder1(), Encoder2(), Decoder1(), Decoder2())

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCELoss(reduction='sum')

        trainer = VAETrainer(model, optimizer, criterion, train_loader, test_loader, LOG_PATH)

        trainer.run(N_EPOCHS, "model", batchSize=1, seed=SEED, checkpointInterval=1, checkpoint=True)

    elif sys.argv[1] == 'test':
        tester()

