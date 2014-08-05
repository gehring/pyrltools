import numpy as np
from scipy.sparse import bsr_matrix

class Projector(object):
    def __init__(self):
        pass

    def __call__(self, state):
        pass

    @property
    def size(self):
        pass

class IdentityProj(Projector):
    def __init__(self, inputSize, **kargs):
        super(IdentityProj, self).__init__()
        self.__size = inputSize

    def __call__(self, state):
        return state

    @property
    def size(self):
        return self.__size

class StateNormalizer(Projector):
    def __init__(self, stateprojector, state_range):
        self.stateprojector = stateprojector
        self.state_range = np.array(state_range)

    def __call__(self, state):
        nstate = (state - self.state_range[0])/(self.state_range[1]-self.state_range[0])
        return self.stateprojector(nstate)

    @property
    def size(self):
        return self.stateprojector.size

class StateNormalizer_Factory(object):
    def __init__(self, stateprojector_factory, **argk):
        self.stateprojector_factory = stateprojector_factory
        self.param =argk

    def __call__(self, **argk):
        new_param = dict(self.param)
        new_param.update([ x for x in argk.items()])
        stateprojector = self.stateprojector_factory(**new_param)
        domain = new_param.get('domain')
        return StateNormalizer(stateprojector, domain.state_range)

class Tiling(object):
    def __init__(self, input_size, ntiles, offset = None):
        self.input_size = input_size
        self.ntiles = ntiles
        self.offset = offset
        if self.offset == None:
            self.offset = np.random.uniform(0, 1, input_size)

    def getIndex(self, inputs):
        index = 0
        for i in range(self.input_size):
            index = (index*self.ntiles +
                        int(inputs[i]*(self.ntiles-1) + self.offset[i]))
        return index

    @property
    def size(self):
        return self.ntiles**self.input_size

class TileCoding(Projector):
    ''' The slowest tilecoding implementation ever'''
    def __init__(self, input_size, ntiles, ntilings, in_range = None):
        super(TileCoding, self).__init__()
        self.__size = ntilings * (ntiles**input_size)
        self.tilings = [Tiling(input_size, ntiles) for i in range(ntilings)]
        self.in_range = in_range

    def __call__(self, state):
        if self.in_range != None:
            state = (state - self.in_range[0])/(self.in_range[1] - self.in_range[0])
        out = np.zeros(self.size)
        offset = 0
        for tiling in self.tilings:
            index = tiling.getIndex(state)
            out[index + offset] = 1.0
            offset += tiling.size
        return out

    @property
    def size(self):
        return self.__size

class RBFCoding(Projector):
    def __init__(self, input_size,  stddev, c = None, nrbfs = 50, in_range = None):
        super(RBFCoding, self).__init__()
        if c == None:
            self.c = np.array([ np.random.uniform(0,1,size = input_size)
                                    for i in xrange(nrbfs)])
        else:
            self.c = c
        self.__size = self.c.shape[0]
        self.in_range = in_range
        self.stddev_inv = 1.0/stddev

    def __call__(self, state):
        if self.in_range != None:
            state = (state - self.in_range[0])/(self.in_range[1] - self.in_range[0])
        diff = (self.c - state)*self.stddev_inv
        return np.exp(-np.sum(diff**2, axis=1))

    @property
    def size(self):
        return self.__size


class StateActionProjector(object):
    def __init__(self):
        pass

    def __call__(self, state, action):
        pass

    @property
    def size(self):
        pass

class Normalizer(StateActionProjector):
    def __init__(self, stateactionprojector, state_range, action_range):
        self.stateactionprojector = stateactionprojector
        self.state_range = np.array(state_range)
        self.action_range = np.array(action_range)

    def __call__(self, state, action):
        nstate = (state - self.state_range[0])/(self.state_range[1]-self.state_range[0])
        naction = (action - self.action_range[0])/(self.action_range[1]-self.action_range[0])
        return self.stateactionprojector(nstate, naction)

    @property
    def size(self):
        return self.stateactionprojector.size

class Normalizer_Factory(object):
    def __init__(self, stateactionprojector_factory, **argk):
        self.stateactionprojector_factory = stateactionprojector_factory
        self.param =argk

    def __call__(self, **argk):
        new_param = dict(self.param)
        new_param.update([ x for x in argk.items()])
        stateactionproj = self.stateactionprojector_factory(**new_param)
        domain = new_param.get('domain')
        return Normalizer(stateactionproj, domain.state_range, domain.action_range )

class TabularAction(StateActionProjector):
    def __init__(self, state_size, num_action, projector = None, actions = None):
        super(TabularAction, self).__init__()
        self.num_action = num_action
        self.actions = actions

        if projector == None:
            self.projector = IdentityProj(state_size)
        else:
            self.projector = projector

    def __call__(self, state, actions):
        p_size = self.projector.size
        proj = np.zeros((len(actions), p_size * self.num_action))
        for i, a in enumerate(actions):
            if self.actions != None and not isinstance(a, int):
                a = np.array(a)
                a = [ j for j, act in enumerate(self.actions) if np.all(act == a)][0]
            proj[i, p_size*a:p_size*(a+1)] = self.projector(state)

        if len(actions) == 1:
            return proj[0,:]
        else:
            return proj

    @property
    def size(self):
        return self.projector.size*self.num_action

class FlatStateAction(StateActionProjector):
    def __init__(self, state_size, action_dim, projector = None):
        super(FlatStateAction, self).__init__()
        if projector == None:
            self.projector = IdentityProj(state_size)
        else:
            self.projector = projector

        self.action_dim = action_dim

    def __call__(self, state, action):
        proj = np.zeros( state.size + self.action_dim)
        proj[:state.size] = state
        proj[state.size:] = action
        return proj

    @property
    def size(self):
        return self.projector.size + self.action_dim

class FlatStateAction_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([ x for x in argk.items()])
        domain = params.get('domain')
        return FlatStateAction(domain.state_dim, domain.action_dim)

