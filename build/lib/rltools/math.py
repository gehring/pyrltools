import numpy as np
from numpy.random import random_sample

def discrete_sample(probabilities, size=1):
    bins = np.cumsum(probabilities)
    if size >1:
        return np.digitize(random_sample(size), bins)
    else:
        return np.digitize(random_sample(size), bins)[0]