import os
from midi import arry2mid
import numpy as np
import torch
from collections import Counter

PATH = "data/midi/"
DEST = "output/"

if not os.path.exists(DEST):
    os.mkdir(DEST)

for root, _, files in os.walk(PATH):
    for f in files:
        stem = f.split(".")[0]
        with open(root + f, "rb") as tp:
            sample = torch.load(tp)
            # sample = sample.numpy()
            # print(Counter(np.reshape(sample, -1)))
            arry2mid(sample.numpy(), DEST +  stem + ".mid", 10000000)
