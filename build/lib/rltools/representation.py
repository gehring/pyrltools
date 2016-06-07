import numpy as np
import scipy.spatial.distance as distance
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

class TabularActionProjector(object):
    def __init__(self, actions, phi):
        self.actions = actions
        self.phi = phi
        self.nactions = len(actions) if not isinstance(actions, int) else actions
        self.size = phi.size*self.nactions
    def __call__(self, state, action = None):
        x = self.phi(state)
        if action is None:
            if issubclass(x.dtype.type, np.uint):
                x = np.vstack(( x + self.phi.size*i for i in xrange(self.nactions)))
                assert(issubclass(x.dtype.type, np.uint))
            else:
                x = np.kron(np.eye(self.nactions), x)#np.diag(x[None,:], self.nactions)
        else:
            a = action
            if not isinstance(action, int):
                a = getActionIndex(action, self.actions)
            if issubclass(x.dtype.type, np.uint):
                x = x + self.phi.size*a
                assert(issubclass(x.dtype.type, np.uint))
            else:
                x = np.pad(x,
                           (a*self.phi.size,
                            self.phi.size*(self.nactions-a-1)),
                            mode = 'constant')
                
        return x.T
    
class StateActionProjection(object):
    def __init__(self, actions, size = None, phi = None):
        self.actions = actions
        self.phi = phi
        if phi is None:
            assert(size is not None)
            assert(type(size) is int)
            self.size = size
        else:
            self.size = phi.size
        
    def __call__(self, state, action = None):
        if self.phi is None:
            if action is None:
                x = np.vstack((np.hstack((state, a)) for a in self.actions))
            else:
                x = np.hstack((state, action))
        else:    
            if action is None:
                x = np.vstack((self.phi(np.hstack((state, a))) for a in self.actions))
            else:
                x = self.phi(np.hstack((state, action)))
            
        return x
    
def getActionIndex(action, actions):
    assert(not isinstance(actions, int))
    return int(np.all(action == actions, axis=1).nonzero()[0][0])


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
        if state.ndim == 1:
            state = state.reshape((1,-1))[:,:,None]
        else:
            state = state[:,:,None]
        
        nstate = (state[:, self.input_index, :] - self.state_range[0][None,:,None])/(self.state_range[1]-self.state_range[0])[None,:,None]
        indicies =((self.offset[None,:,:] + nstate)*self.ntiles[None,:,None]).astype(np.int)
        return self.hashing(indicies) + self.index_offset[None,:]
#         nstate = (state[self.input_index] - self.state_range[0])/(self.state_range[1]-self.state_range[0])
#         indicies =((self.offset + nstate[:,None])*self.ntiles[:,None]).astype(np.uint)
#         return self.hashing.__call__(indicies) + self.index_offset

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

        self.__size  = int(self.__size)


    def __call__(self, state):
        if state.ndim == 1:
            state = state.reshape((1,-1))
            
        if self.bias_term:
            indices = np.hstack(chain((t(state) for t in self.tilings), [np.zeros((state.shape[0], 1), dtype='uint')])) + self.index_offset
        else:
            indices = np.hstack((t(state) for t in self.tilings)) + self.index_offset
        return indices.squeeze()

    @property
    def size(self):
        return self.__size

class TileCodingDense(TileCoding):
    def __init__(self, *args, **kargs):
        super(TileCodingDense, self).__init__(*args, **kargs)

    def __call__(self, state):
        print state.shape
        if state.ndim>1:
            phi= np.zeros((state.shape[0],self.size))
            for i,s in enumerate(state):
                phi[i,super(TileCodingDense, self).__call__(state)] = 1
        else:
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
        self.memory = int(memory)
        for i in range(4):
            self.rndseq = self.rndseq << 8 | np.random.random_integers(np.iinfo('int16').min,
                                                                       np.iinfo('int16').max,
                                                                       16384) & 0xff
    def __call__(self, indices):
        rnd_seq = self.rndseq
        a = self.increment*np.arange(indices.shape[1])
        index = indices + a[None,:,None]
        index = index - (index.astype(np.int)/rnd_seq.size)*rnd_seq.size
        hashed_index = (np.sum(rnd_seq[index], axis=1)).astype(np.int)
        return (hashed_index - (hashed_index/self.memory).astype(np.int)*self.memory).astype('uint')
#         index = np.remainder(indices + self.increment*np.arange(indices.shape[0])[:,None], self.rndseq.size).astype('int')
#         return np.remainder(np.sum(self.rndseq[index], axis=0), self.memory).astype('uint')

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
        self.dim_offset =np.cumprod(np.hstack(([1],self.dims[:0:-1]))).astype('int')[None,::-1,None]

    def __call__(self, indices):
        if self.wrap:
            indices = np.remainder(indices, self.dims[None, :,None])
            #return np.ravel_multi_index(np.remainder(indices.T, self.dims[:,None]), self.dims, mode=).astype('uint')
        else:
            indices = np.clip(indices, 0, self.dims[None, :,None]-1)
            #return np.ravel_multi_index(np.clip(indices.T, 0, self.dims[:,None]-1), self.dims).astype('uint')
        return np.sum(indices*self.dim_offset, axis=1)

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
#         self.RBFs = RBFCoding(stddev, c, **params)
        self.__size = c.shape[0]
        self.c = c.T[None,:,:]
        if stddev.ndim == 1:
            self.w = stddev[None,:,None]
        else:
            self.w = stddev.T[None,:,:]
        
    def __call__(self, state):
#         x = self.RBFs(state)
#         return x/np.sum(x)
        if state.ndim == 1:
            state = state.reshape((1,-1))

        dsqr = -(((state[:,:,None] - self.c)/self.w)**2).sum(axis=1)
        e_x = np.exp(dsqr - dsqr.min(axis=1)[:,None])
        return e_x/ e_x.sum(axis=1)[:,None]


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

# class TabularAction(StateActionProjector):
#     def __init__(self, state_size, num_action, projector = None, actions = None):
#         super(TabularAction, self).__init__()
#         self.num_action = num_action
#         self.actions = actions
# 
#         if projector == None:
#             self.projector = IdentityProj(state_size)
#         else:
#             self.projector = projector
# 
#     def __call__(self, state, actions):
#         p_size = self.projector.size
#         proj = np.zeros((len(actions), p_size * self.num_action))
#         for i, a in enumerate(actions):
#             if self.actions != None and not isinstance(a, int):
#                 a = np.array(a)
#                 a = [ j for j, act in enumerate(self.actions) if np.all(act == a)][0]
#             proj[i, p_size*a:p_size*(a+1)] = self.projector(state)
# 
#         if len(actions) == 1:
#             return proj[0,:]
#         else:
#             return proj
# 
#     @property
#     def size(self):
#         return self.projector.size*self.num_action

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

class Rbf_kernel(object):
    def __init__(self, width = None):
        self.w = width if width is not None else 1

    def __call__(self, X, Y=None):
        X = X/self.w
        if Y is None:
            dist = distance.squareform(distance.pdist(X, 'sqeuclidean'))
        else:
            Y = Y/self.w
            dist = distance.cdist(X, Y, 'sqeuclidean')
        K = np.exp(-dist)
        return K

class Poly_kernel(object):
    def __init__(self, d, c):
        self.c = c
        self.d = d
    def __call__(self, X, Y=None):
        if Y is None:
            dist = X.dot(X.T) + self.c
        else:
            dist = X.dot(Y.T) + self.c
        return dist**self.d

class Partial_Kernel(object):
    def __init__(self, indices, kernel):
        self.indices = indices
        self.kernel = kernel
    def __call__(self, X, Y=None):
        if Y is not None:
            return self.kernel(X[:,self.indices], Y[:,self.indices])
        else:
            return self.kernel(X[:,self.indices])

class Multiplied_kernel(object):
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
    def __call__(self, X, Y=None):
        return self.k1(X,Y) * self.k2(X,Y)