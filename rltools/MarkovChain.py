import numpy as np

class MarkovChain(object):
    def __init__(self, P):
        self.P = P
        self.num_states = P.shape[0]
        self.states = range(self.num_states)

    def start(self, p = None):
        self.s = np.random.choice(self.states, p=p)

    def step(self):
        self.s = np.random.choice(self.states, p=self.P[:, self.s])
        return self.s