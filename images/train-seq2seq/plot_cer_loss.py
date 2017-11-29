import json
import numpy as np
import matplotlib.pyplot as plt

with open("run-seq2seq_best-tag-dev_cer.json") as fid:
    cers = json.load(fid)
    cers = [c[2] for c in cers][4:]

with open("run-seq2seq_best-tag-dev_loss.json") as fid:
    losses = json.load(fid)
    losses = [c[2] for c in losses][4:]

t = range(len(cers))

fig, ax1 = plt.subplots()
ax1.plot(t, losses, 'b-')
ax1.set_ylabel('Loss')
ax1.tick_params('y', colors='b')
ax1.set_ylim([25, 100])

ax2 = ax1.twinx()
ax2.plot(t, cers, 'r-')
ax2.set_ylabel('CER')
ax2.tick_params('y', colors='r')
ax2.set_ylim([0.1, 1.2])

fig.tight_layout()
plt.savefig("loss_cer.svg")
