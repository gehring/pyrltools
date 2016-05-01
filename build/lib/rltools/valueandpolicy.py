import theano
import theano.tensor as T

import numpy as np

from rltools.theanotools import HiddenLayer

from itertools import chain


def build_net(x, n_in, n_out, layers, activations, rng, W = None, b = None):
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
    
    return out_layer.output, list(chain(*[l.params for l in hidden_layer]))

class NeuroValPol(object):
    def __init__(self, 
                 n_input,
                 n_params, 
                 layers,
                 rng, 
                 alpha,
                 gamma,
                 beta_1, 
                 beta_2,
                 input_layer = None,
                 activations = None, 
                 W = None, 
                 b= None):
        
        if activations is None:
            activations = [T.tanh]*len(layers) + [None]
        
        # state at time t
        s = T.vector('s', dtype = theano.config.floatX)
        s.tag.test_value = np.random.rand(2).astype(theano.config.floatX)
        # state at time t+1
        sp = T.vector('sp', dtype = theano.config.floatX)
        sp.tag.test_value = np.random.rand(2).astype(theano.config.floatX)
        
        # policy parameters to use
        p_params = T.matrix('p_params', dtype = theano.config.floatX)
        p_params.tag.test_value = np.random.rand(13, 101).astype(theano.config.floatX)
        # importance sampling for the different parameters
        rho = T.vector('rho', dtype = theano.config.floatX)
        rho.tag.test_value = np.random.rand(13).astype(theano.config.floatX)
        
        
        bs = s.dimshuffle('x',0)
        bsp = sp.dimshuffle('x',0)
        if input_layer is not None:
            # this is where any pre-processing of the state is done
            x, n_input = input_layer(bs)
            xp, _ = input_layer(bsp)
        else:
            x = bs
            xp = bsp
        
        # concatenate the states with the parameters
        inputs = T.concatenate((T.repeat(x, p_params.shape[0], axis=0), p_params), axis=1)
        inputsp = T.concatenate((T.repeat(xp, p_params.shape[0], axis=0), p_params), axis=1)
        
        # reward at time t
        r_in = T.scalar('r', dtype = theano.config.floatX)
        r_in.tag.test_value = np.random.rand(1).astype(theano.config.floatX)[0]
        r = r_in.dimshuffle(('x'))
        
        # terminal state flag
        isterminal = T.printing.Print('term')(T.scalar('isterminal', dtype = 'int8'))
        isterminal.tag.test_value = np.random.rand(1).astype('int8')[0]
#         broadcast_isterminal = isterminal.dimshuffle(('x'))
        
        # build the network (symbolically). Allocates parameters if needed
        self.out, self.params = build_net(inputs, 
                                             n_input+n_params, 
                                             1, 
                                             layers, 
                                             activations, 
                                             rng, 
                                             W, 
                                             b)
        # build the network for the next state input. Re-uses parameters
        self.outp, _ = build_net(inputsp, 
                                     n_input+n_params, 
                                     1, 
                                     layers, 
                                     activations, 
                                     rng, 
                                     self.params[::2], 
                                     self.params[1::2])
        
        # learning parameters
        self._alpha = theano.shared(np.array( alpha, dtype=theano.config.floatX), 'alpha', allow_downcast = True, borrow=False)
        self._beta_1 = theano.shared(np.array( beta_1, dtype=theano.config.floatX), 'beta_1', allow_downcast = True, borrow=False)
        self._beta_2 = theano.shared(np.array( beta_2, dtype=theano.config.floatX), 'beta_2', allow_downcast = True, borrow=False)
        
        # regularization on the input weights for each node
        L2_reg = sum([ (p**2).sum() for p in self.params[::2]])
        L1_reg = sum([ abs(p).sum() for p in self.params[::2]])
        
        # define the cost function
        next_val = T.switch(isterminal, T.zeros_like(self.outp), self.outp)
        
        cost = ((rho.dimshuffle((0,'x'))*(r + gamma*next_val-self.out))**2).sum()  \
                    + L2_reg*self._beta_2 + L1_reg*self._beta_1
                    
        # define the graident with respect to the cost. Note that we do
        # not propagate the gradient through next_val
        gparams = [T.grad(cost, 
                          param,
                          consider_constant = [next_val, self.outp]) 
                    for param in self.params]
        
        # define the update rule
        updates = [ (param, param -self._alpha * gparam)
                    for (param, gparam) in zip(self.params, gparams)]           
        
        # compile the update function
        self.__update_function = theano.function(
            inputs = [s, r_in, sp, p_params, rho, isterminal],
            outputs = cost,
            updates = updates,
            allow_input_downcast=True
            )
        
        # compile the evaluate function
        self.__evaluate = theano.function([s, p_params], 
                                          self.out,
                                            allow_input_downcast= True)
        
        self.__evaluate_grad = theano.function([s, p_params], 
                                          (self.out, T.grad(self.out[0,0], p_params)),
                                            allow_input_downcast= True)
        
    def __call__(self, state, policy_param):
        if policy_param.ndim < 2:
            policy_param = policy_param[None,:]
            
        return self.__evaluate(state, policy_param)
    
    def evluate_with_gradient(self, state, policy_param):
        if policy_param.ndim < 2:
            policy_param = policy_param[None,:]
            
        return self.__evaluate_grad(state, policy_param)
    
    def update(self, s_t, r_t, s_tp1, policy_param, rho):
        if policy_param.ndim < 2:
            policy_param = policy_param[None,:]
            rho = np.array([rho])
        term = np.int8(s_tp1 is None)
        if s_tp1 is None:
            s_tp1 = np.empty_like(s_t)
        return self.__update_function(s_t, r_t, s_tp1, policy_param, rho, term)
        
        
    @property
    def alpha(self):
        return self._alpha.get_value()
    
    @alpha.setter
    def alpha(self, value):
        self._alpha.set_value(value)
        
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
        
        
class LinearPolicy(object):
    def __init__(self, 
                 n_input, 
                 n_output, 
                 input_layer = None):
        self.phi = input_layer
        self.n_input = n_input
        self.n_output = n_output
        
    def __call__(self, state, params):
        if self.phi is not None:
            state = self.phi(state)
           
        if params.ndim == 1:
            params = params[None,:]           
        W = params[:,:-self.n_output].reshape((-1, self.n_input, self.n_output)) 
        b = params[:,-self.n_output:]
        
        return (state.dot(W) + b).squeeze()
    
    def get_gaussian_rho(self, state, behaviour, params, sigma):
        
        x = behaviour[None,:]
        output = self.__call__(state, params)

        if output.ndim == 1:
            if params.ndim == 1:
                output = output.reshape((1,-1))
            else:
                output = output.reshape((-1,1))
        
        if not isinstance(sigma, float):
            sigma = sigma[None,:]
        
        diff = ((x - output)**2)/(2*sigma**2)
        pdf = np.exp(-diff.sum(axis=1))
        pdf /= pdf[0]
        return pdf
            