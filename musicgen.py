import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import sys
import random
import os
import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import numpy as np

from util.Trainer import Trainer
from midi96 import samples_to_midi

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()
torch.set_printoptions(threshold=5000)

MIDI_MAT = 96 * 96
N_MEASURES = 16
N_EPOCHS = 200
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
        return X

class Encoder1(nn.Module):
    """
    Pre-encoder of the network, one is created for each measure.
    """

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(MIDI_MAT, 2000),
            nn.ReLU(),
            nn.Linear(2000, 200),
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
            nn.Linear(200, 2000),
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

        self.encoder1 = torch.nn.ModuleList([encoder1 for _ in range(N_MEASURES)])
        self.encoder2 = encoder2
        self.decoder1 = decoder1
        self.decoder2 = torch.nn.ModuleList([decoder2 for _ in range(N_MEASURES)])
    
    def forward(self, x0):
        mus = []
        sigmas = []
        x4 = torch.empty((len(x0), 16, MIDI_MAT)).to(DEVICE)

        for i in range(len(x0)):
            x1 = torch.empty((3200)).to(DEVICE)
            for j in range(N_MEASURES):
                x1[j*200:(j+1)*200] = self.encoder1[j](x0[i, j])

            mu, sigma = self.encoder2(x1)
            mus.append(mu)
            sigmas.append(sigma)

            std = torch.exp(sigma / 2)
            eps = torch.randn_like(std)
            x2 = eps.mul(std).add_(mu)

            x3 = self.decoder1(x2)

            for k in range(N_MEASURES):
                x4[i, k] = self.decoder2[k](x3[k*200:(k+1)*200])

        return x4, torch.stack(mus), torch.stack(sigmas)

    def decoder(self, sample):
        midi_array = torch.empty((16, MIDI_MAT))

        x = self.decoder1(sample)
        for i in range(N_MEASURES):
            midi_array[i] = self.decoder2[i](x[i*200:(i+1)*200])
        return midi_array
    
class VAETrainer(Trainer):
    def __init__(self, model, optimizer, criterion, trainloader, testloader, logPath):
        super().__init__(model, optimizer, criterion, trainloader, testloader, logPath)

    def calc_loss(self, x):
        x = x.view(-1, 16, MIDI_MAT)
        y, mu, sigma = self.model(x)
  
        
        reconLoss = self.crit(y, x)

        KLLoss = 0.5 * \
            torch.sum(torch.exp(sigma) + mu*mu - 1.0 - sigma)

        return reconLoss + KLLoss

def tester():
    tf = transforms.ToTensor()
    data = MidiDataset("data/midi/", tf)

    train_set, test_set = torch.utils.data.random_split(data, [20000, 5000])

    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)

    model = VAE(Encoder1(), Encoder2(), Decoder1(), Decoder2())

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss(reduction='sum')
    trainer = VAETrainer(model, optimizer, criterion, train_loader, test_loader, LOG_PATH)

    trainer._load_checkpoint("model")
    output = trainer.model.decoder(torch.randn((120)).to(DEVICE))
    print(output)

    samples_to_midi(output.detach().reshape((16, 96, 96)), "epoch1.mid", thresh=0.5)


if __name__ == "__main__":

    if sys.argv[1] == 'train':
        tf = transforms.ToTensor()
        data = MidiDataset("data/midi/", tf)

        data_len = len(os.listdir("data/midi/"))
        train_len = math.floor(0.8 * data_len)
        test_len = data_len - train_len

        train_set, test_set = torch.utils.data.random_split(data, [train_len, test_len])

        train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)

        model = VAE(Encoder1(), Encoder2(), Decoder1(), Decoder2())

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCELoss(reduction='sum')

        trainer = VAETrainer(model, optimizer, criterion, train_loader, test_loader, LOG_PATH)

        trainer.run(N_EPOCHS, "model", batchSize=5, seed=SEED, checkpointInterval=20, checkpoint=False)

    elif sys.argv[1] == 'test':
        tester()
