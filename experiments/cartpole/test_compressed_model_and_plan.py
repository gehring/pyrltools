# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:48:11 2016

@author: cgehri
"""
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
from rltools.npplanning import sample_gaussian
from rltools.planner import SingleEmbeddedAgent, CompressedLEMAgent

state_range = Cartpole.state_range

def get_XAXtp1_samples(sample_traj, rew_fn, ter_fn):
    X_t = []
    A_t = []
    X_tp1 = []
    X_term = []
    A_term = []
    R_t = []
    R_term = []
    for s_t, a, s_tp1 in chain(*sample_traj):
        if s_tp1 is None or ter_fn(s_tp1):
            X_term.append(s_t)
            R_term.append(rew_fn(s_t, s_tp1))
            A_term.append(a)
        else:
            X_t.append(s_t)
            R_t.append(rew_fn(s_t, s_tp1))
            A_t.append(a)
            X_tp1.append(s_tp1)
    return (np.array(X_t), np.array(A_t), np.array(R_t), np.array(X_tp1),
            np.array(X_term), np.array(A_term), np.array(R_term))
        
        

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
    
def rew_cartpole(s_t, s_tp1):
    angle = np.abs(np.pi - s_tp1[1])
    angle = np.pi - angle if angle > np.pi else angle
    return np.exp( -((angle)**2 + (s_tp1[3]*0.5)**2))*(1-np.min([np.abs(s_tp1[0])*0.1, 1]))
    
def term_cartpole(s_t):
    angle = np.abs(np.pi - s_t[1])
    angle = np.pi - angle if angle > np.pi else angle
    return angle< np.pi/12 and np.abs(s_t[3]) < 0.5

np.random.seed(10)

width = np.array([0.3, 0.1, 0.1, 0.05])
scale = ((state_range[1] - state_range[0]) * width)

num_gauss = 2000
w = sample_gaussian(state_range[0].shape[0], num_gauss, scale)   
phi = lambda X: fourier_features(X, w)

filename = 'cartpole-test-2.data'
with open(filename, 'rb') as f:
    sample_traj = pickle.load(f)

#(parsed_samples, term_samples, term_rew_samples), actions = parse_data_discrete_actions(sample_traj, rew_cartpole, term_cartpole)
#Xa_t, Ra, Xa_tp1 = zip(*parsed_samples)
#Xa_term, Ra_term = zip(*term_samples)
X_t, A_t, R_t, X_tp1, X_term, A_term, R_term = get_XAXtp1_samples(sample_traj, rew_cartpole, term_cartpole)
actions = np.array([np.array([-3.]),
          np.array([0.]),
          np.array([3.])])
          
def single_action_kernel(i):
    return lambda b: np.exp(-np.sum(((actions[i][None,:]-b)/6.0)**2, axis=1)/(0.5**2))
    #return lambda b: (actions[i][None,:] == b).astype('float').squeeze()

action_kernels = [ single_action_kernel(i) for i in xrange(len(actions))]
#action_kernels = [ lambda b: a[None,:] == b for a in actions]

#print [a[None,:] == actions for a in actions]
#print action_kernels[0](actions)
print [a_k(actions) for a_k in action_kernels]
#SingleEmbeddedAgent
#CompressedLEMAgent
agent = SingleEmbeddedAgent(plan_horizon = 100,
                 dim = num_gauss*2, 
                 X_t = X_t,
                 A_t = A_t, 
                 X_tp1 = X_tp1, 
                 R_t = R_t,
                 X_term = X_term,
                 A_term = A_term,
                 R_term = R_term,
                 max_rank = 300,
                 blend_coeff = 0.2,
                 phi = phi,
                 action_kernels = action_kernels,
                 actions = actions,
                 learning_rate = 0.01,
                 discount = 0.99,
                 update_models = False,
                 use_valuefn = False,
                 use_diff_model = True)

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
x_tp1 = domain.reset()[1]
x_t, a_t, r_t = [None]*3
traj = []                
for i in xrange(300):
    print 'step'
    if x_t is not None and x_tp1 is not None:
        r_t = rew_cartpole(x_t, x_tp1)
    agent.step(x_t, a_t, r_t, x_tp1)
    x_t = x_tp1
    a_t = agent.get_action()
    r_t, x_tp1 = domain.step(a_t)
    traj.append(x_t)
    if x_tp1 is None:
        break
                      
with open('scompressLEM-test-traj.data', 'wb') as f:
    pickle.dump(traj, f)          
                      
                      
"""
plan_horizon,
                 dim, 
                 num_actions, 
                 Xa_t, 
                 Xa_tp1, 
                 Ra,
                 Xa_term,
                 Ra_term,
                 max_rank,
                 blend_coeff,
                 phi,
                 update_models = False):"""                      
                      