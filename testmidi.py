from util.midi import samples_to_midi
import os
import torch
import time

PATH = "data/"

for root, _, files in os.walk(PATH):
    for f in files:
        stem = f.split(".tp")[0]
        with open(root+f, "rb") as tp:
            midi_array = torch.load(tp)
