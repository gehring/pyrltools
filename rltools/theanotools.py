""" Code taken from theano's tutorial on MLPs """

import numpy
import numpy as np

import time

import theano
import theano.tensor as T
import theano.sparse

from itertools import chain

import sklearn.cross_validation
from sklearn import preprocessing


def sym_NRBF(x, wrt, dx, centers, widths, bias_term):
        size = centers.shape[0]
        centers = centers.T
        
        dx0 = dx
        
        x = x.dimshuffle((0,1,'x'))
        dx = dx.dimshuffle((0,1,'x'))
        x = T.patternbroadcast(x, (False,False,True))
        dx = T.patternbroadcast(dx, (False,False,True))
        if widths.ndim > 1:
            widths = widths.T
            w = theano.shared(widths.reshape((1,widths.shape[0], -1)),
                         borrow=False,
                         broadcastable = (True,False, False))
        else:
            w = theano.shared(widths.reshape((1,widths.shape[0], -1)),
                         borrow = False,
                         broadcastable = (True,False, True))
        c = theano.shared(centers.reshape((1,centers.shape[0], -1)), borrow=False, broadcastable = (True,False, False))

        dsqr = -(((x - c)/w)**2).sum(axis=1, keepdims=False)
        e_x = T.exp(dsqr - dsqr.min(axis=1, keepdims=True))
        out = e_x / e_x.sum(axis=1, keepdims=True)

        if bias_term:
            out = T.concatenate((out, T.ones((out.shape[0], 1))), axis=1)
            size += 1
            
        return out, T.Rop(out, wrt, dx0), size
    
def sym_RBF(x, wrt, dx, centers, widths, bias_term):
        size = centers.shape[0]
        centers = centers.T
        
        dx0 = dx
        
        x = x.dimshuffle((0,1,'x'))
        dx = dx.dimshuffle((0,1,'x'))
        x = T.patternbroadcast(x, (False,False,True))
        dx = T.patternbroadcast(dx, (False,False,True))
        if widths.ndim > 1:
            widths = widths.T
            w = theano.shared(widths.reshape((1,widths.shape[0], -1)),
                         borrow=False,
                         broadcastable = (True,False, False))
        else:
            w = theano.shared(widths.reshape((1,widths.shape[0], -1)),
                         borrow = False,
                         broadcastable = (True,False, True))
        c = theano.shared(centers.reshape((1,centers.shape[0], -1)), borrow=False, broadcastable = (True,False, False))

        dsqr = -(((x - c)/w)**2).sum(axis=1, keepdims=False)
        out = T.exp(dsqr)

        if bias_term:
            out = T.concatenate((out, T.ones((out.shape[0], 1))), axis=1)
            size += 1
        return out, T.Rop(out, wrt, dx0), size
    
def sym_NeuroSFTD(x, wrt, dx, n_in, n_out, layers, activations, rng, W = None, b = None):
    
    # build all layers
    
    #if no initial value given, set to none
    if W is None:
        w0 = None
    else:
        w0 = W[0]
        
    if b is None:
        b0 = None
    else:
        b0 = b[0]
    
    # first layer
    layers = layers + [n_out]
#     n_outputs = layers[1:] + [n_out]
    hidden_layer = [HiddenLayer(rng, x, n_in, layers[0], w0, b0, activations[0])]
    
    # remaining layers
    for i,l in enumerate(layers[:-1]):
        if W is None:
            w0 = None
        else:
            w0 = W[i+1]
            
        if b is None:
            b0 = None
        else:
            b0 = b[i+1]
        hidden_layer.append(HiddenLayer(rng, hidden_layer[-1].output, layers[i], layers[i+1], w0, b0, activations[i+1]))
        
    out_layer = hidden_layer[-1]
    # return ouput and directional derivative (jacobian times a vector)
    return out_layer.output, T.Rop(out_layer.output, wrt, dx), list(chain(*[l.params for l in hidden_layer]))
#     return out_layer.output, out_layer.output, list(chain(*[l.params for l in hidden_layer]))

def sym_QNeuroSFTD(x, wrt, dx, n_in, n_out, n_NN, layers, activations, rng, W = None, b = None):
    
    # build all layers
    
    #if no initial value given, set to none
    if W is None:
        w0 = None
    else:
        w0 = W[0]
        
    if b is None:
        b0 = None
    else:
        b0 = b[0]
    
    # first layer
    layers = layers + [n_out]
#     n_outputs = layers[1:] + [n_out]
    hidden_layer = [shared_input_QHiddenLayer(rng, x, n_in, layers[0], n_NN, w0, b0, activations[0])]
    
    # remaining layers
    for i,l in enumerate(layers[:-1]):
        if W is None:
            w0 = None
        else:
            w0 = W[i+1]
            
        if b is None:
            b0 = None
        else:
            b0 = b[i+1]
        hidden_layer.append(QHiddenLayer(rng, hidden_layer[-1].output, layers[i], layers[i+1], n_NN, w0, b0, activations[i+1]))
        
    out_layer = hidden_layer[-1]
    # return ouput and directional derivative (jacobian times a vector)
    return out_layer.output, T.Rop(out_layer.output, wrt, dx), list(chain(*[l.params for l in hidden_layer]))
#     return out_layer.output, out_layer.output, list(chain(*[l.params for l in hidden_layer]))

class NeuroSFTD(object):
    def __init__(self, 
                 n_input, 
                 layers,
                 rng, 
                 alpha,
                 alpha_mu, 
                 eta, 
                 beta_1, 
                 beta_2,
                 input_layer = None,
                 activations = None, 
                 W = None, 
                 b= None):
        
        
#         ds = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,False))('ds')
        s = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,False))('s')
        sp = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,False))('sp')
        ds = sp-s
        
        
        s.tag.test_value = np.random.rand(5, 2).astype('float32')
#         ds.tag.test_value = np.random.rand(5, 2).astype('float32')
        sp.tag.test_value = np.random.rand(5, 2).astype('float32')

        if input_layer is not None:
            x,_, n_input = input_layer(s,ds)
            xp,_, _ = input_layer(sp,ds)
        else:
            x = s
            xp = sp
        
#         y = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,True))('y')
        r = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,True))('r')
        isterminal = T.TensorType(dtype = 'int8', broadcastable=(False,True))('isterminal')
        
#         y.tag.test_value = numpy.random.rand(5).astype('float32')[:,None]
        r.tag.test_value = numpy.random.rand(5).astype('float32')[:,None]
        isterminal.tag.test_value = numpy.random.rand(5).astype('int8')[:,None]
        
        if activations is None:
            activations = [T.tanh]*len(layers) + [None]
            
        self.out, self.dout, self.params = sym_NeuroSFTD(x, s, ds, n_input, 1, layers, activations, rng, W, b)
        self.outp, self.doutp, _ = sym_NeuroSFTD(xp, sp, ds, n_input, 1, layers, activations, rng, self.params[::2], self.params[1::2])
        
        next_val = T.switch(isterminal, T.zeros_like(self.outp), self.outp)
        
        self._alpha = theano.shared(np.array( alpha, dtype=theano.config.floatX), 'alpha', allow_downcast = True, borrow=False)
        self._eta = theano.shared(np.array( eta, dtype=theano.config.floatX), 'eta', allow_downcast = True, borrow=False)
        self._beta_1 = theano.shared(np.array( beta_1, dtype=theano.config.floatX), 'beta_1', allow_downcast = True, borrow=False)
        self._beta_2 = theano.shared(np.array( beta_2, dtype=theano.config.floatX), 'beta_2', allow_downcast = True, borrow=False)
        
        self.alpha_mu = alpha_mu
        self.mu = None
        
        L2_reg = sum([ (p**2).sum() for p in self.params[::2]])
        L1_reg = sum([ abs(p).sum() for p in self.params[::2]])

        jac_cost = T.switch(isterminal, 0.0, (((self.dout - r)**2).sum()))
#         cost = (1-self._eta)*(((y-self.out)**2).sum()) + self._eta*(((self.dout - r)**2).sum()) \
#                     + L2_reg*self._beta_2 + L1_reg*self._beta_1
                    
        cost = (1-self._eta)*(((r + next_val-self.out)**2).sum()) + self._eta*jac_cost \
                    + L2_reg*self._beta_2 + L1_reg*self._beta_1
                    
        gparams = [T.grad(cost, 
                          param,
                          consider_constant = [isterminal, next_val]) 
                    for param in self.params]
        
        updates = [ (param, param - self._alpha * gparam)
                    for (param, gparam) in zip(self.params, gparams)]

        self.__update_function = theano.function(
            inputs = [s, r, sp, isterminal],
            outputs = cost,
            updates = updates,
            allow_input_downcast=True
            )
        
        self.__evaluate = theano.function([s], self.out, allow_input_downcast= True)
        
    def __call__(self, state):
        if state is None:
            return 0
        else:
            if state.ndim == 1:
                state = state[None,:]
            return self.__evaluate(state)
    
    def update(self, s_t, r, s_tp1):
        if s_t is None:
            return
        
        if s_tp1 is None:
            s_tp1 = np.zeros_like(s_t)
            ter = np.ones(1, dtype='int8')
        else:
            ter = np.zeros(1, dtype='int8')
            
        
        if self.mu is None:
            self.mu = r
        
        r = r - self.mu
        self.__update_function(s_t[None,:], np.array([r]).reshape((1,1)), s_tp1[None,:], ter[:,None])
        self.mu += self.alpha_mu * r
        
        
    @property
    def alpha(self):
        return self._alpha.get_value()
    
    @alpha.setter
    def alpha(self, value):
        self._alpha.set_value(value)
        
    @property
    def eta(self):
        return self._eta.get_value()
    
    @eta.setter
    def eta(self, value):
        self._eta.set_value(value)
        
    @property
    def beta_1(self):
        return self._beta_1.get_value()
    
    @beta_1.setter
    def beta_1(self, value):
        self._beta_1.set_value(value)
        
    @property
    def beta_2(self):
        return self._beta_2.get_value()
    
    @beta_2.setter
    def beta_2(self, value):
        self._beta_2.set_value(value)
        
class QNeuroSFTD(object):
    def __init__(self, 
                 n_input,
                 n_actions, 
                 layers,
                 rng, 
                 alpha,
                 alpha_mu, 
                 eta, 
                 beta_1, 
                 beta_2,
                 input_layer = None,
                 activations = None, 
                 W = None, 
                 b= None):
        
        
        self.n_actions = n_actions
        
#         ds = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,False))('ds')
        s = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,False))('s')
        a_t = T.TensorType(dtype = 'int32', broadcastable=(False,))('a_t')
        sp = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,False))('sp')
        a_tp1 = T.TensorType(dtype = 'int32', broadcastable=(False,))('a_tp1')
        
        ds = sp-s
        
        
        s.tag.test_value = np.random.rand(5, 2).astype('float32')
        a_t.tag.test_value = np.random.randint(0,3, 5).astype('int32')
#         ds.tag.test_value = np.random.rand(5, 2).astype('float32')
        sp.tag.test_value = np.random.rand(5, 2).astype('float32')
        a_tp1.tag.test_value = np.random.randint(0,3, 5).astype('int32')

        if input_layer is not None:
            x,_, n_input = input_layer(s,ds)
            xp,_, _ = input_layer(sp,ds)
        else:
            x = s
            xp = sp
        
#         x = x.dimshuffle(('x',0,1))
#         x = T.repeat(x, n_actions, axis=0)
#         x = T.tile(x, (n_actions, 1,1))
#         xp = xp.dimshuffle(('x',0,1))
#         xp = T.tile(xp, (n_actions, 1,1))
#         xp = T.repeat(xp, n_actions, axis=0)
        
#         y = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,True))('y')
        r = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,True))('r')
        isterminal = T.TensorType(dtype = 'int8', broadcastable=(False,True))('isterminal')
        
#         y.tag.test_value = numpy.random.rand(5).astype('float32')[:,None]
        r.tag.test_value = numpy.random.rand(5).astype('float32')[:,None]
        isterminal.tag.test_value = numpy.random.rand(5).astype('int8')[:,None]
        
        if activations is None:
            activations = [T.tanh]*len(layers) + [None]
        
            
        self.out, self.dout, self.params = sym_QNeuroSFTD(x, s, ds, n_input, 1, n_actions, layers, activations, rng, W, b)
        self.outp, self.doutp, _ = sym_QNeuroSFTD(xp, sp, ds, n_input, 1, n_actions, layers, activations, rng, self.params[::2], self.params[1::2])
        
        next_val = T.switch(isterminal, T.zeros((s.shape[0], 1)), self.outp[a_tp1,:,:])
        
        self._alpha = theano.shared(np.array( alpha, dtype=theano.config.floatX), 'alpha', allow_downcast = True, borrow=False)
        self._eta = theano.shared(np.array( eta, dtype=theano.config.floatX), 'eta', allow_downcast = True, borrow=False)
        self._beta_1 = theano.shared(np.array( beta_1, dtype=theano.config.floatX), 'beta_1', allow_downcast = True, borrow=False)
        self._beta_2 = theano.shared(np.array( beta_2, dtype=theano.config.floatX), 'beta_2', allow_downcast = True, borrow=False)
        
        self.alpha_mu = alpha_mu
        self.mu = None
        
        L2_reg = sum([ (p**2).sum() for p in self.params[::2]])
        L1_reg = sum([ abs(p).sum() for p in self.params[::2]])

#         cost = (1-self._eta)*(((y-self.out)**2).sum()) + self._eta*(((self.dout - r)**2).sum()) \
#                     + L2_reg*self._beta_2 + L1_reg*self._beta_1
                    
        cost = (1-self._eta)*(((r + next_val-self.out[a_t,:,:])**2).sum()) + self._eta*(((self.dout[a_t,:,:] - r)**2).sum()) \
                    + L2_reg*self._beta_2 + L1_reg*self._beta_1
                    
        gparams = [T.grad(cost, 
                          param,
                          consider_constant = [isterminal, next_val]) 
                    for param in self.params]

        updates = [ (param, param - self._alpha * gparam)
                    for (param, gparam) in zip(self.params, gparams)]
        self.__update_function = theano.function(
            inputs = [s, a_t, r, sp, a_tp1, isterminal],
            outputs = cost,
            updates = updates,
            allow_input_downcast=True
            )
        
        self.__evaluate = theano.function([s], self.out, allow_input_downcast= True)
        self.__evaluate_action = theano.function([s,a_t], self.out[a_t,:,:], allow_input_downcast= True)
        
    def __call__(self, state, action=None):
        if state is None:
            if action is None:
                return np.zeros(self.n_actions)
            else:
                return 0
        else:
            if state.ndim == 1:
                    state = state[None,:]
            if action is None:
                return self.__evaluate(state).squeeze()
            else:
                a = np.zeros((1,))
                a[0] = action
                return self.__evaluate_action(state, a).squeeze()
    
    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t is None:
            return
        
        a = np.zeros((2,1))
        a[0,0] = a_t
        
#         if s_tp1 is None:
#             v_tp1 = 0
#         else:
#             v_tp1 = self.__call__(s_tp1, a_tp1)
        
        if s_tp1 is None:
            s_tp1 = np.zeros_like(s_t)
            ter = np.ones(1, dtype='int8')
        else:
            a[1,0] = a_tp1
            ter = np.zeros(1, dtype='int8')
            
        
        if self.mu is None:
            self.mu = np.float(r)
        
        
        r = r - self.mu
        
#         v_t = self.__call__(s_t,a_t)
        
            
        a_t = a[0,:]
        a_tp1 = a[1,:]
        
#         delta = r + 0.99*v_tp1 - v_t
#         print s_t[None,:].shape, a_t.shape, np.array([r]).reshape((1,1)).shape, s_tp1[None,:].shape, a_tp1.shape, ter[:,None].shape
        d=self.__update_function(s_t[None,:], a_t, np.array([r]).reshape((1,1)), s_tp1[None,:], a_tp1, ter[:,None])
        

        
#         my_up = np.zeros_like(d[1])
#         my_up[a_t[0]] = -delta*d[2].T*2
#         diff = d[1] - my_up
#         
#         no_up = d[1].copy()
#         no_up[a_t[0]] = 0.0
#         no_up = np.linalg.norm(no_up)
#         
#         my_up[a_t[0]] = 0.0
#         my_up = np.linalg.norm(my_up)
#         if  np.linalg.norm(diff)>0:
#             print  np.linalg.norm(diff)
#         print delta
#         print np.linalg.norm(self.params[0].get_value())
#         print d[1].shape, diff.shape, np.abs(d[0]-delta**2), np.linalg.norm(diff)/np.linalg.norm(d[1]), no_up, my_up
        self.mu += self.alpha_mu * r
        
        
    @property
    def alpha(self):
        return self._alpha.get_value()
    
    @alpha.setter
    def alpha(self, value):
        self._alpha.set_value(value)
        
    @property
    def eta(self):
        return self._eta.get_value()
    
    @eta.setter
    def eta(self, value):
        self._eta.set_value(value)
        
    @property
    def beta_1(self):
        return self._beta_1.get_value()
    
    @beta_1.setter
    def beta_1(self, value):
        self._beta_1.set_value(value)
        
    @property
    def beta_2(self):
        return self._beta_2.get_value()
    
    @beta_2.setter
    def beta_2(self, value):
        self._beta_2.set_value(value)
        
        
        
class Theano_RBF_stateaction(object):
    def __init__(self, centers, widths, bias_term = True, normalized = False):
        S = T.TensorType(dtype = theano.config.floatX, broadcastable = (False, False))('S')
        A = T.TensorType(dtype = theano.config.floatX, broadcastable = (False, False))('A')
        X = T.concatenate((S,A), axis=1)
        dS = T.TensorType(dtype = theano.config.floatX, broadcastable = (False, False))('dS')
        
        if normalized:
            out, dout, self.size = sym_NRBF(X, S, dS, centers, widths, bias_term)
        else:
            out, dout, self.size = sym_RBF(X, S, dS, centers, widths, bias_term)
            
        self.proj = theano.function([S, A], out, allow_input_downcast=True)
        self.doutdx = theano.function([S, A, dS], dout, allow_input_downcast=True)


    def __call__(self, S, A):
        if S.ndim == 1:
            S = S.reshape((1,-1))
            if A.ndim == 1:
                A = A.reshape((1,-1))
            else:
                S = np.repeat(S, A.shape[0], axis=0)
        phis = self.proj(S, A)
        return phis

    def getdphids(self, S, A, dS):
        if dS.ndim == 1:
            dS = dS.reshape((1,-1))
        if S.ndim == 1:
            S = S.reshape((1,-1))
            if A.ndim == 1:
                A = A.reshape((1,-1))
            else:
                S = np.repeat(S, A.shape[0], axis=0)
        dphids = self.doutdx(S, A, dS)
        return dphids
    
class Theano_RBF_Projector(object):
    def __init__(self, centers, widths, bias_term = True, normalized = False):
        x = T.TensorType(dtype = theano.config.floatX, broadcastable = (False, False))('x')
        x.tag.test_value = np.random.rand(10,2).astype(theano.config.floatX)
        
        dx = T.TensorType(dtype = theano.config.floatX, broadcastable = (False, False))('dx')
        dx.tag.test_value = np.random.rand(10,2).astype(theano.config.floatX)
        
        if normalized:
            out, dout, self.size = sym_NRBF(x, x, dx, centers, widths, bias_term)
        else:
            out, dout, self.size = sym_RBF(x, x, dx, centers, widths, bias_term)
        self.proj = theano.function([x], out, allow_input_downcast=True)
        self.doutdx = theano.function([x,dx], dout, allow_input_downcast=True)
    
    def __call__(self, state):
        if state.ndim == 1:
            phis = self.proj(state[None,:])[0,:]
        else:
            phis = self.proj(state)
        return phis

    def getdphids(self, state, ds):
        if ds.ndim == 1:
            ds = ds.reshape((1,-1))
        if state.ndim == 1:
            state = state.reshape((1,-1))
        dphids = self.doutdx(state, ds)
        return dphids

def sym_tiling_index(X,
                     input_index,
                     ntiles,
                     ntilings,
                     state_range,
                     offset = None,
                     hashing = None):
        s_range = [state_range[0][input_index].copy(), state_range[1][input_index].copy()]
        s_range[0] -= (s_range[1]-s_range[0])/(ntiles-1)

        if isinstance(ntiles, int):
            ntiles = np.array([ntiles]*len(input_index), dtype='uint')

        if offset == None:
            offset = np.empty((ntiles.shape[0], ntilings))
            for i in xrange(ntiles.shape[0]):
                offset[i,:] = -np.linspace(0, 1.0/ntiles[i], ntilings, False);
        if hashing == None:
            hashing = Theano_IdentityHash(ntiles)

        input_index = np.array(input_index, dtype='uint')
        size = ntilings*(hashing.memory)
        index_offset = (hashing.memory * np.arange(ntilings)).astype('int')


        nX = (X[:,input_index,:] - s_range[0][None,:,None])/(s_range[1]-s_range[0])[None,:,None]
        indices = T.cast(((offset[None,:,:] + nX)*ntiles[None,:,None]), 'int32')
        hashed_index = hashing.getHashedFunction(indices) + index_offset[None,:]
        return hashed_index, size
    


class Theano_IdentityHash(object):
    def __init__(self, dims):
        self.dims = dims
        self.memory = np.prod(dims)

    def getHashedFunction(self, indices):
        dims = np.cumprod(np.hstack(([1],self.dims[:0:-1]))).astype('int')[None,::-1,None]
        return T.sum(indices*dims, axis=1, keepdims = False)

class Theano_UNH(object):
    increment = 470
    def __init__(self, input_size, memory):
        self.rndseq = np.zeros(16384, dtype='int')
        self.input_size = input_size
        self.memory = memory
        for i in range(4):
            self.rndseq = self.rndseq << 8 | np.random.random_integers(np.iinfo('int16').min,
                                                                       np.iinfo('int16').max,
                                                                       16384) & 0xff

    def getHashedFunction(self, indices):
        rnd_seq = theano.shared(self.rndseq, borrow=False)
        a = self.increment*np.arange(self.input_size)
        index = indices + a[None,:,None]
        index = index - (T.cast(index, 'int64')/self.rndseq.size)*self.rndseq.size
        hashed_index = T.cast(T.sum(rnd_seq[index], axis=1, keepdims = False), 'int64')
        return hashed_index - (hashed_index/(int(self.memory)))*int(self.memory)


class Theano_Tiling(object):
    def __init__(self,
                 input_indicies,
                 ntiles,
                 ntilings,
                 hashing,
                 state_range,
                 bias_term = True):
        if hashing == None:
            hashing = [None]*len(ntilings)
        X = T.TensorType(dtype = theano.config.floatX, broadcastable = (False, False, True))('X')
        X.tag.test_value = numpy.random.rand(1, 2,1).astype('float32')
        tilings, sizes = zip(*[sym_tiling_index(X, in_index, nt, t, state_range, hashing = h,)
                   for in_index, nt, t, h in zip(input_indicies, ntiles, ntilings, hashing)])

        self.__size = int(sum(sizes))
        index_offset = np.zeros(len(ntilings), dtype = 'int')
        index_offset[1:] = np.cumsum(sizes)
        index_offset = np.hstack( [np.array([off]*t, dtype='int')
                                            for off, t in zip(index_offset, ntilings)])

        all_indices = T.cast(T.concatenate(tilings, axis=1), 'int32') + index_offset.astype('int')
        if bias_term:
            all_indices = T.cast(T.concatenate((all_indices, self.__size*T.ones((all_indices.shape[0], 1))), axis=1), 'int32')
            self.__size += 1

        self.proj = theano.function([X], all_indices, allow_input_downcast=True)

    def __call__(self, state):
        if state.ndim == 1:
#             state = state.reshape((1,-1,1))
            phi = self.proj(state[None,:,None])[0,:]
        else:
            phi = self.proj(state[:,:,None])
        return phi

    @property
    def size(self):
        return self.__size




class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.n_out = n_out

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
#
#         W.tag.test_value = W.get_value()
#         b.tag.test_value = b.get_value()
#         input.tag.test_value = np.random.rand(60, n_in).astype(theano.config.floatX)
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
        
class shared_input_QHiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, n_NN, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.n_out = n_out

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_NN, n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_NN, 1, n_out), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', broadcastable = (False, True, False), borrow=True)

        self.W = W
        self.b = b
#
#         print input.tag.test_value.shape,W.get_value().shape, b.get_value().shape
        batch_dot = theano.map(fn = lambda w, x: T.dot(x, w),
                               sequences = [self.W],
                               non_sequences = [input])
        lin_output = batch_dot[0] + self.b
#         print lin_output.tag.test_value.shape
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class QHiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, n_NN, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.n_out = n_out

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_NN, n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_NN, 1, n_out), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', broadcastable = (False, True, False), borrow=True)

        self.W = W
        self.b = b
#
#         print input.tag.test_value.shape,W.get_value().shape, b.get_value().shape
#         batch_dot = theano.map(fn = lambda i, x, w: T.dot(x[i,:,:], w[i,:,:]),
#                                sequences = [T.arange(n_NN)],
#                                non_sequences = [input, self.W])
#         lin_output = batch_dot[0] + self.b
        lin_output = T.batched_dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.input = input
        if isinstance(n_hidden, int):
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh
            )

            # The logistic regression layer gets as input the hidden units
            # of the hidden layer
            self.outputlayer = HiddenLayer(
                rng=rng,
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out,
                activation=None
            )
            # end-snippet-2 start-snippet-3
            # L1 norm ; one regularization option is to enforce L1 norm to
            # be small
            self.L1 = (
                abs(self.hiddenLayer.W).sum()
                + abs(self.outputlayer.W).sum()
            )

            # square of L2 norm ; one regularization option is to enforce
            # square of L2 norm to be small
            self.L2_sqr = (
                (self.hiddenLayer.W ** 2).sum()
                + (self.outputlayer.W ** 2).sum()
            )

            # the parameters of the model are the parameters of the two layer it is
            # made out of
            self.params = self.hiddenLayer.params + self.outputlayer.params
        else:
            self.hiddenLayer = [HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden[0],
                activation=T.tanh
            )]
            for n in n_hidden[1:]:
                self.hiddenLayer.append(HiddenLayer(
                rng=rng,
                input=self.hiddenLayer[-1].output,
                n_in=self.hiddenLayer[-1].n_out,
                n_out=n,
                activation=T.tanh
            ))
                # The logistic regression layer gets as input the hidden units
            # of the hidden layer
            self.outputlayer = HiddenLayer(
                rng=rng,
                input=self.hiddenLayer[-1].output,
                n_in=n_hidden[-1],
                n_out=n_out,
                activation=None
            )
            # L1 norm ; one regularization option is to enforce L1 norm to
            # be small
            self.L1 = (
                sum([ abs(h.W).sum() for h in self.hiddenLayer])
                + abs(self.outputlayer.W).sum()
            )

            # square of L2 norm ; one regularization option is to enforce
            # square of L2 norm to be small
            self.L2_sqr = (
                sum([ (h.W ** 2).sum() for h in self.hiddenLayer])
                + (self.outputlayer.W ** 2).sum()
            )

            # the parameters of the model are the parameters of the two layer it is
            # made out of
            self.params = list(chain(*[h.params for h in self.hiddenLayer])) + self.outputlayer.params



        self.output = self.outputlayer.output

    def L2cost(self, y):
        err = self.output[:,0] - y
        return T.mean((err*err))

    def getf(self, x):
        return theano.function([x], self.output)

def MLPregression(learning_rate,
                    mlp,
                    l2_coeff,
                    l1_coeff,
                    samples,
                    target,
                    batch_size,
                    validate_ratio,
                    rng,
                    n_epochs= 100,
                    validation_freq = 2,
                    patience = 10000,
                    patience_increase = 2,
                    improvement_thresh = 0.995):



    train_samples, test_samples, train_target, test_target = \
        sklearn.cross_validation.train_test_split(samples,
                                              target,
                                              test_size = validate_ratio,
                                              random_state = rng
                                              )
    train_samples = theano.shared(train_samples.astype(theano.config.floatX),borrow = False)
    test_samples = theano.shared(test_samples.astype(theano.config.floatX),borrow = False)
    train_target = theano.shared(train_target.astype(theano.config.floatX),borrow = False)
    test_target = theano.shared(test_target.astype(theano.config.floatX),borrow = False)


    n_train_batches = train_samples.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_samples.get_value(borrow=True).shape[0] / batch_size

    x = mlp.input
    y = T.fvector('y')
#     y.tag.test_value = numpy.random.rand(batch_size*2).astype(theano.config.floatX)
    index = T.lscalar()
    index.tag.test_value = 0

    cost = (mlp.L2cost(y) + mlp.L1*l1_coeff + mlp.L2_sqr*l2_coeff)

    test_model = theano.function(
            inputs=[index],
            outputs=mlp.L2cost(y),
            givens={
                x: test_samples[index * batch_size:(index + 1) * batch_size],
                y: test_target[index * batch_size:(index + 1) * batch_size]
            }
        )

    gparams = [T.grad(cost, param) for param in mlp.params]

    updates = [ (param, param - learning_rate * gparam)
                    for (param, gparam) in zip(mlp.params, gparams)]

    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: train_samples[index * batch_size:(index + 1) * batch_size],
            y: train_target[index * batch_size:(index + 1) * batch_size]
            }
        )

    epoch = 0
    count = 0

    best_val = numpy.inf

    best_params = [ p.get_value(borrow = False) for p in mlp.params]
    done_train = False

#     start_time = time.clock()
    while epoch < n_epochs and (not done_train):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_cost = train_model(minibatch_index)

            if count %validation_freq == 0:
                val_cost = np.mean([test_model(i) for i in xrange(n_test_batches)])

                if val_cost < best_val:
                    if val_cost < best_val*improvement_thresh:
                        patience = max(patience, count*patience_increase)
                    best_val = val_cost
#                     print best_val
                    best_params = [ p.get_value(borrow = False) for p in mlp.params]


            # number of minibatches seen
            count += 1

            # termination condition
            done_train = patience <= count

#     end_time = time.clock()
#     print 'Time taken for opt: ' + str(end_time - start_time) + 's'
    for p, best_p in zip(mlp.params, best_params):
        p.set_value(best_p)


    train_samples.set_value([[]])
    train_target.set_value([])
    test_samples.set_value([[]])
    test_target.set_value([])
    return mlp.getf(x)

def MLPSparseRegression(learning_rate,
                    mlp,
                    l2_coeff,
                    l1_coeff,
                    samples,
                    target,
                    batch_size,
                    validate_ratio,
                    rng,
                    n_epochs= 100,
                    validation_freq = 2,
                    patience = 10000,
                    patience_increase = 2,
                    improvement_thresh = 0.995):

    scaler = preprocessing.StandardScaler.fit(samples)
    samples = scaler.transform(samples)

    train_samples, test_samples, train_target, test_target = \
        sklearn.cross_validation.train_test_split(samples,
                                              target,
                                              test_size = validate_ratio,
                                              random_state = rng
                                              )
    train_samples = theano.sparse.shared(train_samples.astype(theano.config.floatX),borrow = False)
    test_samples = theano.sparse.shared(test_samples.astype(theano.config.floatX),borrow = False)
    train_target = theano.shared(train_target.astype(theano.config.floatX),borrow = False)
    test_target = theano.shared(test_target.astype(theano.config.floatX),borrow = False)


    n_train_batches = train_samples.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_samples.get_value(borrow=True).shape[0] / batch_size

    x = mlp.input
    y = T.fvector('y')
#     y.tag.test_value = numpy.random.rand(batch_size*2).astype(theano.config.floatX)
    index = T.lscalar()
    index.tag.test_value = 0

    cost = (mlp.L2cost(y) + mlp.L1*l1_coeff + mlp.L2_sqr*l2_coeff)

    test_model = theano.function(
            inputs=[index],
            outputs=mlp.L2cost(y),
            givens={
                x: test_samples[index * batch_size:(index + 1) * batch_size],
                y: test_target[index * batch_size:(index + 1) * batch_size]
            }
        )

    gparams = [T.grad(cost, param) for param in mlp.params]

    updates = [ (param, param - learning_rate * gparam)
                    for (param, gparam) in zip(mlp.params, gparams)]

    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: train_samples[index * batch_size:(index + 1) * batch_size],
            y: train_target[index * batch_size:(index + 1) * batch_size]
            }
        )

    epoch = 0
    count = 0

    best_val = numpy.inf

    best_params = [ p.get_value(borrow = False) for p in mlp.params]
    done_train = False

#     start_time = time.clock()
    while epoch < n_epochs and (not done_train):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_cost = train_model(minibatch_index)

            if count %validation_freq == 0:
                val_cost = np.mean([test_model(i) for i in xrange(n_test_batches)])

                if val_cost < best_val:
                    if val_cost < best_val*improvement_thresh:
                        patience = max(patience, count*patience_increase)
                    best_val = val_cost
#                     print best_val
                    best_params = [ p.get_value(borrow = False) for p in mlp.params]


            # number of minibatches seen
            count += 1

            # termination condition
            done_train = patience <= count

#     end_time = time.clock()
#     print 'Time taken for opt: ' + str(end_time - start_time) + 's'
    for p, best_p in zip(mlp.params, best_params):
        p.set_value(best_p)


    train_samples.set_value([[]])
    train_target.set_value([])
    test_samples.set_value([[]])
    test_target.set_value([])
    return lambda x: mlp.getf(scaler.transform(x))


def MLPfit(learning_rate,
            n_hidden,
            rng,
            l2_coeff,
            l1_coeff,
            samples,
            target,
            batch_size = 100,
            validate_ratio=0.1,
            n_epochs= 1000,
            validation_freq = 2,
            patience = 10000,
            patience_increase = 2,
            improvement_thresh = 0.995):
    x = T.matrix('x')
#     x.tag.test_value =  numpy.random.rand(batch_size*2,samples.shape[1]).astype(theano.config.floatX)
    mlp = MLP(rng,
              x,
              samples.shape[1],
              n_hidden,
              1)
    return MLPregression(learning_rate,
                         mlp,
                         l2_coeff,
                         l1_coeff,
                         samples,
                         target,
                         batch_size,
                         validate_ratio,
                         rng,
                         n_epochs,
                         validation_freq,
                         patience,
                         patience_increase,
                         improvement_thresh)

def MLPSparsefit(learning_rate,
            n_hidden,
            rng,
            l2_coeff,
            l1_coeff,
            samples,
            target,
            batch_size = 100,
            validate_ratio=0.1,
            n_epochs= 1000,
            validation_freq = 2,
            patience = 10000,
            patience_increase = 2,
            improvement_thresh = 0.995):
    x = theano.sparse.matrix('x')
#     x.tag.test_value =  numpy.random.rand(batch_size*2,samples.shape[1]).astype(theano.config.floatX)
    mlp = MLP(rng,
              x,
              samples.shape[1],
              n_hidden,
              1)
    return MLPregression(learning_rate,
                         mlp,
                         l2_coeff,
                         l1_coeff,
                         samples,
                         target,
                         batch_size,
                         validate_ratio,
                         rng,
                         n_epochs,
                         validation_freq,
                         patience,
                         patience_increase,
                         improvement_thresh)




