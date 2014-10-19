import numpy as np
from itertools import chain

class Tiling(object):
    def __init__(self,
                 input_index,
                 ntiles,
                 ntilings,
                 state_range,
                 offset = None):

        self.state_range = [state_range[0][input_index].copy(), state_range[1][input_index].copy()]
        self.state_range[0] -= (self.state_range[1]-self.state_range[0])/(ntiles-1)

        self.offset = offset
        if offset == None:
            self.offset = -np.linspace(0, 1.0/ntiles, ntilings, False);
        self.dims = np.array([ntiles]*len(input_index), dtype='int32')
        self.input_index = np.array(input_index, dtype='int32')
        self.size = ntilings*(ntiles**len(self.input_index))
        self.index_offset = ntiles**len(self.input_index) * np.arange(ntilings)
        self.ntiles = ntiles

    def __call__(self, state):
        proj_state = np.zeros(self.size, dtype = 'int32')
        proj_state[self.getIndices(state)] = 1
        return proj_state

    def getIndices(self, state):
        nstate = (state[self.input_index] - self.state_range[0])/(self.state_range[1]-self.state_range[0])
        indicies = np.clip(((self.offset[None,:] + nstate[:,None])*self.ntiles).astype(int), 0, self.ntiles-1)
        return np.ravel_multi_index(indicies, self.dims) + self.index_offset

class TileCoding(object):
    def __init__(self,
                 input_indicies,
                 ntiles,
                 ntilings,
                 state_range,
                 bias_term = True):
        self.state_range = state_range
        self.tilings = [Tiling(in_index, nt, t, state_range)
                        for in_index, nt, t in zip(input_indicies, ntiles, ntilings)]
        self.__size = sum(map(lambda x: x.size, self.tilings))
        self.bias_term = bias_term
        if bias_term:
            self.__size += 1

    def __call__(self, state):
        if self.bias_term:
            return np.hstack(chain((t(state) for t in self.tilings), [np.array(1, dtype='int32')]))
        else:
            return np.hstack((t(state) for t in self.tilings))

    @property
    def size(self):
        return self.__size


