"""
Results from DS2 https://arxiv.org/abs/1512.02595

VoxForge American-Canadian 
VoxForge Commonwealth
VoxForge European 
VoxForge Indian
CHiME eval clean 
CHiME eval noise 
"""

import matplotlib.pyplot as plt
import numpy as np

results = [[7.55, 4.85], [13.56, 8.15], [17.55, 12.76],
           [22.44, 22.15], [3.34, 3.46], [21.79, 11.84]]
model, humans = zip(*results)

fig, ax = plt.subplots(figsize=(10, 3))
x = np.arange(len(results))
ax.bar(x-0.14, model, width=0.28, color='b', align='center')
ax.bar(x+0.14, humans, width=0.28, color='g', align='center')
plt.savefig("human_model.svg")
