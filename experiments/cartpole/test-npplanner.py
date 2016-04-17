import __builtin__

try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile

import numpy as np
import pickle
from itertools import chain
from rltools.cartpole import Cartpole
from rltools.npplanning import sparse_build_np_models, build_np_models, find_stoc_plan, build_approx_gauss_models, sample_gaussian, convert_compressed_to_embed

state_range = Cartpole.state_range

def parse_data_discrete_actions(sample_traj, rew_fn, ter_fn):
    data = list(chain(*sample_traj))
    samples = dict()
    ter_samples = dict()
    
    for s_t, a, s_tp1 in data:
        a.flags.writeable = False
        if not a.data in samples:
            samples[a.data] = []
        if not a.data in ter_samples:
            ter_samples[a.data] = []
            
        if ter_fn(s_tp1):
            ter_samples[a.data].append((s_t, rew_fn(s_t, s_tp1), s_tp1))
        else:
            samples[a.data].append( (s_t, rew_fn(s_t, s_tp1), s_tp1))
        
    discrete_actions = samples.keys()
    parsed_samples = [ zip(*samples[a]) for a in discrete_actions]
    parsed_ter_samples = [ zip(*ter_samples[a]) for a in discrete_actions]
    
    parsed_samples = [ (np.array(X), np.array(R), np.array(Xp)) for X,R,Xp in parsed_samples ]
    term_rew_samples = (parsed_samples[0][0], np.zeros_like(parsed_samples[0][1]))

    term_samples = []
    for i in xrange(len(discrete_actions)):
        if len(parsed_ter_samples[i]) == 0:
            term_samples.append( (parsed_samples[0][0], np.zeros_like(parsed_samples[0][1])))
        else:
            term_samples.append((np.array(parsed_ter_samples[i][0]), np.array(parsed_ter_samples[i][1])))
            
    #term_samples = [ (np.array(X), np.array(R)) for X,R,Xp in parsed_ter_samples ]  
    
    
    return (parsed_samples, term_samples, term_rew_samples), [ np.frombuffer(a) for a in discrete_actions]
    

def fourier_features(X, w):
    if X.ndim == 1:
        X = X.reshape((1,-1))
    features = np.hstack((X, np.zeros((X.shape[0], 1)))).dot(w)        
    features = np.hstack((np.sin(features), np.cos(features)))
    return features.squeeze()/ np.sqrt(w.shape[1])

def kernel(X, Y):
    if X.ndim == 1:
        X = X.reshape((1,-1))
        
    if Y.ndim == 1:
        Y = Y.reshape((1,-1))
    width = np.array([0.3, 0.1, 0.1, 0.05])
    scale = ((state_range[1] - state_range[0]) * width)[None,:,None]
        

    # first compute the difference between states
    diff = X[:,:,None] - Y.T[None,:,:]
    
    # wrap around the angles
    angles = np.abs(diff[:,1,:])
    wrap_around = angles > np.pi 
    diff[:,1,:] = wrap_around * (np.pi*2 - angles) + (1-wrap_around)*angles
    
    # get the squared distance
    dsqr = -((diff/scale)**2).sum(axis=1)
    
    return np.exp(dsqr).squeeze()


@profile
def cat(sample_traj, rew_fn, ter_fn, plan_length = 200, lamb=0.1, gamma = 0.5, forward = True, models = None):
    
    (trans_samples, ter_samples, ter_rew_samples), actions = parse_data_discrete_actions(sample_traj, rew_fn, ter_fn)
    
    sparse = False
    approx_np = True
    
    if models is None:
        if approx_np:
            width = np.array([0.3, 0.1, 0.1, 0.05])
            scale = ((state_range[1] - state_range[0]) * width)
            
            num_gauss = 5000
            w = sample_gaussian(state_range[0].shape[0], num_gauss, scale)   
            phi = lambda X: fourier_features(X, w)
            models = build_approx_gauss_models(scale, 
                                  trans_samples, 
                                  ter_samples, 
                                  num_gauss = num_gauss,
                                  phi = phi,
                                  k = 300)
            models = convert_compressed_to_embed(*models)
                                  
        else:    
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
                                 
    print 'model computed'
    cartpole = Cartpole(m_c = 1,
                 m_p = 1,
                 l = 1,
                 g = 9.81,
                 x_damp = 0.1,
                 theta_damp = 0.1,)
    
    cartpole.action_range[0][:] = -3
    cartpole.action_range[1][:] = 3
    cartpole.dt[-1] =  1.0/20

    domain = cartpole
    domain.random_start = False
    domain.start_state[:] = [0.0, 0.2, 0.0, 0.0]
    state = domain.reset()[1]
    plan, alphas, betas = find_stoc_plan(state, plan_length, len(actions), models, gamma, forward = forward, sparse = sparse, approx_np = approx_np)
    print 'done'
    # plan[:,:] = np.array([0,0,1])[None,:]
    
    traj_open = [state.copy()]
    traj = [state.copy()]
    s_t = state
    for p in plan:
        a = actions[np.argmax(p)]
#        print np.argmax(p), p.max()
        r_t, s_tp1 = domain.step(a)
        if s_tp1 is None:
            break
        else:
            traj_open.append( s_tp1)
        s_t = s_tp1
        
    state = domain.reset()[1]
    for i in xrange(300):
        a = actions[np.argmax(plan[0])]
#        print np.argmax(p), p.max()
        r_t, s_tp1 = domain.step(a)
        if s_tp1 is None:
            break
        else:
            traj.append( s_tp1)
        plan += 1.0
        plan /= np.sum(plan, axis=1)[:,None]
        print 'start'
        plan, alphas, betas = find_stoc_plan(s_tp1, plan_length, len(actions), models, gamma, forward = forward, sparse = sparse, plan = plan, approx_np = approx_np)
        print 'done'
        s_t = s_tp1
        
    return traj, models, plan
      
      
def rew_cartpole(s_t, s_tp1):
    angle = np.abs(np.pi - s_tp1[1])
    angle = np.pi - angle if angle > np.pi else angle
    return np.exp( -((angle)**2 + (s_tp1[3]*0.5)**2))*(1-np.min([np.abs(s_tp1[0])*0.1, 1]))
    
def term_cartpole(s_t):
    angle = np.abs(np.pi - s_t[1])
    angle = np.pi - angle if angle > np.pi else angle
    return angle< np.pi/12 and np.abs(s_t[3]) < 0.5
    
    
filename = 'cartpole-test-gus.data'
with open(filename, 'rb') as f:
    sample_traj = pickle.load(f)

models = None
traj, models, plan = cat(sample_traj = sample_traj,
           rew_fn = rew_cartpole,
           ter_fn = term_cartpole,
           plan_length = 100,
           lamb = .2,
           gamma = 0.0,
           forward = True)


      