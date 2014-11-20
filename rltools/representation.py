import numpy as np
from itertools import chain

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

class TabularState(Projector):
    def __init__(self, number_of_states, **kargs):
        super(TabularState, self).__init__()
        self.__size = number_of_states

    def __call__(self, state):
        state_v = np.zeros((self.size, 1), dtype='int32')
        state_v[state] = 1
        return state_v

    @property
    def size(self):
        return self.__size

class Tiling(object):
    def __init__(self,
                 input_index,
                 ntiles,
                 ntilings,
                 state_range,
                 offset = None,
                 hashing = None):

        self.state_range = [state_range[0][input_index].copy(), state_range[1][input_index].copy()]
        self.state_range[0] -= (self.state_range[1]-self.state_range[0])/(ntiles-1)


        self.hashing = hashing

        if isinstance(ntiles, int):
            ntiles = np.array([ntiles]*len(input_index), dtype='uint')

        self.offset = offset
        if offset == None:
            self.offset = np.empty((ntiles.shape[0], ntilings))
            for i in xrange(ntiles.shape[0]):
                self.offset[i,:] = -np.linspace(0, 1.0/ntiles[i], ntilings, False);

        if self.hashing == None:
            self.hashing = IdentityHash(ntiles)

        self.input_index = np.array(input_index, dtype='uint')
        self.size = ntilings*(self.hashing.memory)
        self.index_offset = (self.hashing.memory * np.arange(ntilings)).astype('uint')
        self.ntiles = ntiles


    def __call__(self, state):
        return self.getIndices(state)

    def getIndices(self, state):
        nstate = (state[self.input_index] - self.state_range[0])/(self.state_range[1]-self.state_range[0])
        indicies =((self.offset + nstate[:,None])*self.ntiles[:,None]).astype(np.uint)
        return self.hashing.__call__(indicies) + self.index_offset

class TileCoding(Projector):
    def __init__(self,
                 input_indicies,
                 ntiles,
                 ntilings,
                 hashing,
                 state_range,
                 bias_term = True):
        super(TileCoding, self).__init__()
        if hashing == None:
            hashing = [None]*len(ntilings)
        self.state_range = state_range
        self.tilings = [Tiling(in_index, nt, t, state_range, hashing = h)
                        for in_index, nt, t, h in zip(input_indicies, ntiles, ntilings, hashing)]
        self.__size = sum(map(lambda x: x.size, self.tilings))
        self.bias_term = bias_term
        self.index_offset = np.zeros(len(ntilings), dtype = 'uint')
        self.index_offset[1:] = np.cumsum(map(lambda x: x.size, self.tilings[:-1]))
        self.index_offset = np.hstack( [np.array([off]*t, dtype='uint')
                                            for off, t in zip(self.index_offset, ntilings)])

        if bias_term:
            self.index_offset = np.hstack((self.index_offset, np.array(self.__size, dtype='uint')))
            self.__size += 1


    def __call__(self, state):
        if self.bias_term:
            return np.hstack(chain((t(state) for t in self.tilings), [np.array(0, dtype='uint')])) + self.index_offset
        else:
            return np.hstack((t(state) for t in self.tilings)) + self.index_offset

    @property
    def size(self):
        return self.__size

class TileCodingDense(TileCoding):
    def __init__(self, *args, **kargs):
        super(TileCodingDense, self).__init__(*args, **kargs)

    def __call__(self, state):
        phi = np.zeros(self.size)
        phi[super(TileCodingDense, self).__call__(state)] = 1
        return phi


class Hashing(object):
    def __init__(self, **kargs):
        pass
    def __call__(self, indices):
        pass


class UNH(Hashing):
    increment = 470
    def __init__(self, memory):
        super(UNH, self).__init__()
        self.rndseq = np.zeros(16384, dtype='int')
        self.memory = memory
        for i in range(4):
            self.rndseq = self.rndseq << 8 | np.random.random_integers(np.iinfo('int16').min,
                                                                       np.iinfo('int16').max,
                                                                       16384) & 0xff
    def __call__(self, indices):
        index = np.remainder(indices + self.increment*np.arange(indices.shape[0])[:,None], self.rndseq.size).astype('int')
        return np.remainder(np.sum(self.rndseq[index], axis=0), self.memory).astype('uint')
    
class PythonHash(Hashing):
    def __init__(self, memory):
        super(PythonHash, self).__init__()
        self.memory = memory

    def __call__(self, indices):
        indices = np.copy(indices, order = 'F')
        indices.flags.writeable = False
        return np.array([np.remainder(hash(indices[:,i].data) , self.memory)
                         for i in xrange(indices.shape[1])], dtype='uint')

class IdentityHash(Hashing):
    def __init__(self, dims, wrap = False):
        super(IdentityHash, self).__init__()
        self.memory = np.prod(dims)
        self.dims = dims.astype('uint')
        self.wrap = wrap

    def __call__(self, indices):
        if self.wrap:
            return np.ravel_multi_index(np.remainder(indices, self.dims[:,None]), self.dims).astype('uint')
        else:
            return np.ravel_multi_index(np.clip(indices, 0, self.dims[:,None]-1), self.dims).astype('uint')


class RBFCoding(Projector):
    def __init__(self,  stddev, c, **params):
        super(RBFCoding, self).__init__()
        self.c = c
        self.__size = self.c.shape[0]
        self.stddev_inv = 1.0/stddev

    def __call__(self, state):
        diff = (self.c - state)*self.stddev_inv
        return np.exp(-np.sum(diff**2, axis=1))

    @property
    def size(self):
        return self.__size


class NRBFCoding(Projector):
    def __init__(self, stddev, c, **params):
        super(NRBFCoding, self).__init__()
        self.RBFs = RBFCoding(stddev, c, **params)

    def __call__(self, state):
        x = self.RBFs(state)
        return x/np.sum(x)

    @property
    def size(self):
        return self.RBFs.size


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

class Concatenator(Projector):
    def __init__(self, projectors, **params):
        super(Concatenator, self).__init__()
        self.projectors = projectors
        self.__size = sum((proj.size for proj in projectors))

    def __call__(self, state):
        return np.hstack(( proj(state) for proj in self.projectors))

    @property
    def size(self):
        return self.__size

class Concatenator_Factory(object):
    def __init__(self, projector_factories, **argks):
        self.projector_factories = projector_factories
        self.params = argks
    def __call__(self, **argk):
        params = dict(self.params)
        params.update([ x for x in argk.items()])
        return Concatenator(projectors = [f(**params) for f in self.projector_factories],
                       **self.params)

class Indexer(Projector):
    def __init__(self, projector, indices, **params):
        super(Indexer, self).__init__()
        self.projector = projector
        self.indices = indices

    def __call__(self, state):
        return self.projector(state[self.indices])

    @property
    def size(self):
        return self.projector.size

class Indexer_Factory(object):
    def __init__(self, projector_factory, **argks):
        self.projector_factory = projector_factory
        self.params = argks
    def __call__(self, **argk):
        params = dict(self.params)
        params.update([ x for x in argk.items()])
        return Indexer(projector = self.projector_factory(**params), **self.params)

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

