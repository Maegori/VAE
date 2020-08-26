from util.midi import midi_to_samples
import os
import math
import numpy as np
import torch
from mido import MidiFile

N_NOTES = 88
N_TICKS = 96 * 16
N_MEASURES = 16
PATH = "MidiFiles"
DEST = "data/"

if not os.path.exists(DEST):
    os.mkdir(DEST)

patterns = dict()
all_samples = []
all_lens = []

for root, _, files in os.walk(PATH):
    for f in files:
        stem = f.split(".mid")[0]
        p = root + "\\" + f
        if not (p.endswith(".mid") or p.endswith(".midi")):
            continue
        try:
            samples = midi_to_samples(root + f)
        except OSError:
            print(f"Corrupt file, skipping {root+f}")
            continue
        if len(samples) < 16:
            continue
        for i in range(math.floor(len(samples)/16)):
            midi_array =np.empty((16, 96, 96), dtype=np.float32)
            for j in range(16):
                midi_array[j] = samples[i+j]
            with open(DEST + stem + str(i) + ".tp", "wb") as tp:
                torch.save(torch.from_numpy(midi_array), tp)
                print(f"saved {DEST + stem + str(i) + '.tp'}")
            

                
        