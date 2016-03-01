import __builtin__

try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile

import numpy as np
from itertools import chain
from rltools.cartpole import Cartpole

state_range = Cartpole.state_range

def parse_data_discrete_actions(sample_traj, rew_fn):
    data = list(chain(*sample_traj))
    samples = dict()
    
    for s_t, a, s_tp1 in data:
        a.flags_writeable = False
        if not a.data in samples:
            samples[a.data] = []
        samples[a.data].append( (s_t, rew_fn(s_t, s_tp1), s_tp1))
        
    discrete_actions = samples.keys()
    parsed_samples = [ zip(*samples[a]) for a in discrete_actions]
    
    
    return zip(*parsed_samples), [ np.frombuffer(a) for a in discrete_actions]
    

def kernel(X, Y):
    if X.ndim == 1:
        X = X.reshape((1,-1))
        
    if Y.ndim == 1:
        Y = Y.reshape((1,-1))
    width = 0.02
    scale = ((state_range[1] - state_range[0]) * width)[None,:,None]
        
    # compute squared weighted distance distance 
    dsqr = -(((X[:,:,None] - Y.T[None,:,:])/scale)**2).sum(axis=1)
    return np.exp(dsqr).squeeze()


@profile
def cat(lamb=0.1, gamma = 0.5, forward = True, sample_traj, rew_fn):
    
    
    sparse = True
    if sparse:
        models = sparse_build_np_models(kernel, 
                                 trans_samples, 
                                 ter_samples, 
                                 ter_rew_samples, 
                                 lamb)
    else:
        models = build_np_models(kernel, 
                             trans_samples, 
                             ter_samples, 
                             ter_rew_samples, 
                             lamb)    
    
    
    Xa = models[-1] 
    domain.random_start = False
    state = domain.reset()
    plan, alphas, betas = find_stoc_plan(state, 200, 3, models, gamma, forward = forward, sparse = sparse)
    # plan[:,:] = np.array([0,0,1])[None,:]
    
    traj = []
    actions = domain.discrete_actions
    s_t = state
    for p in plan:
        a = actions[np.argmax(p)]
#        print np.argmax(p), p.max()
        r_t, s_tp1 = domain.step(a)
        if s_tp1 is None:
            break
        else:
            traj.append((s_t, s_tp1))
        s_t = s_tp1
      
      