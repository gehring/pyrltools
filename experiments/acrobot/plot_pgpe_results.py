import matplotlib.pyplot as plt
import pickle

import scipy.stats as stats
import numpy as np

def mean_confidence_interval(data, confidence=0.95):
    a = data
    n = a.shape[0]
    m, se = np.mean(a, axis=0), stats.sem(a, axis=0)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, h



fname = 'pgpe_results2.data'

with open(fname, 'rb') as f:
    results = pickle.load(f)


print zip(*results)[0]
scores = np.array(zip(*results)[1])
print scores[:,-1]
y, yerr = mean_confidence_interval(scores, confidence = 0.95)
plt.errorbar(np.arange(y.shape[0]), y, yerr )
plt.show()