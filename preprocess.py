import midi
import os
import math
import numpy as np
import torch
from mido import MidiFile

N_NOTES = 88
N_TICKS = 96 * 16
N_MEASURES = 16
PATH = "MIDI files/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]/0"
DEST = "data/midi/"

if not os.path.exists(DEST):
    os.mkdir(DEST)

patterns = dict()
all_samples = []
all_lens = []

for root, _, files in os.walk(PATH):
    for f in files:
        stem = f.split(".")[0]
        p = root + "\\" + f
        if not (p.endswith(".mid") or p.endswith(".midi")):
            continue

        samples = np.array(midi.mid2arry(MidiFile(p)), dtype=np.float32)
        
        i = 0
        for i in range(math.floor(samples.shape[0] / N_TICKS)):
            midi_array = samples[i * N_TICKS:(i+1)*N_TICKS,:]
            with open(DEST + stem + str(i) + ".tp", "wb") as tp:
                torch.save(torch.from_numpy(midi_array), tp)
                print(f"saved {DEST + stem + str(i) + '.tp'}")


                
        