"""
GMM-HMM - http://www.fit.vutbr.cz/research/groups/speech/publi/2013/vesely_interspeech2013_IS131333.pdf
human-level, https://arxiv.org/pdf/1610.05256.pdf

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.368.3047&rep=rep1&type=pdf
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/FeatureEngineeringInCD-DNN-ASRU2011-pub.pdf
https://36e9b848-a-62cb3a1a-s-sites.googlegroups.com/site/tsainath/DistHF.pdf
http://www.cs.toronto.edu/~asamir/papers/icassp13_cnn.pdf, convolution and depth
http://www.mirlab.org/conference_papers/International_Conference/ICASSP%202014/papers/p5609-soltau.pdf
https://arxiv.org/pdf/1505.05899.pdf, IBM
https://arxiv.org/pdf/1604.08242v2.pdf, IBM
https://arxiv.org/pdf/1609.03528v1.pdf, Microsoft
https://arxiv.org/abs/1610.05256, Microsoft
https://arxiv.org/pdf/1703.02136.pdf, IBM
https://arxiv.org/abs/1708.06073, Microsoft

Human
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

hmm = 18.6
humans = 5.9
models = [[datetime.date(2011, 8, 1), 16.1],
          [datetime.date(2011, 12, 1), 15.5],
          [datetime.date(2012, 9, 1), 13.3],
          [datetime.date(2013, 5, 1), 11.5],
          [datetime.date(2014, 5, 1), 10.4],
          [datetime.date(2015, 5, 1), 8.0],
          [datetime.date(2016, 7, 1), 6.6],
          [datetime.date(2016, 9, 1), 6.3],
          [datetime.date(2016, 10, 1), 5.9],
          [datetime.date(2017, 3, 1), 5.5],
          [datetime.date(2017, 8, 1), 5.1]]

dates, vals = zip(*models)
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(dates, vals, 'k--')
ax.plot(dates, vals, 'k.', markersize=8)

datemin = datetime.date(2011, 1, 1)
datemax = datetime.date(2018, 1, 1)
ax.set_xlim(datemin, datemax)
ax.plot([datemin, datemax], [hmm]*2, 'b--')
ax.plot([datemin, datemax], [humans]*2, 'r--')

plt.savefig("wer.svg")

