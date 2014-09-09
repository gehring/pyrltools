import numpy as np

class RDG(object):

    def __init__(self, num_graphs, num_nodes, num_obs, **kargs):
        self.transitions = np.dstack(( np.random.randint(0, num_nodes,
                                                         (num_obs, num_nodes))
                                       for i in xrange(num_graphs)))
        self.states = np.zeros(num_graphs, dtype='int32')
        self.dummy_indices = np.arange(num_graphs)
        self.num_nodes = num_nodes

    def reset(self):
        self.states[:] = 0

    def update(self, obs_index):
        self.states = self.transitions[obs_index, self.states, self.dummy_indices]
        return self.getState()

    def getState(self):
        # get a vector with ones in the indices of the occupied node
        # and zero everywhere else
        s = self.states + (self.num_nodes ** self.dummy_indices)
        s[0] -= 1
        obs = np.zeros(dtype = 'int32')
        obs[self.states] = 1
        return obs

    def copy(self):
        newrdg = RDG(1, 1, 1)
        newrdg.transitions = self.transitions.copy()
        newrdg.states = self.states.copy()
        newrdg.dummy_indices = self.dummy_indices.copy()
        newrdg.num_nodes = self.num_nodes
        return newrdg