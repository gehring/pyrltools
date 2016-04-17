# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:16:12 2016

@author: cgehri
"""
from itertools import izip
from scipy.linalg import lu_solve

import numpy as np

from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.npplanning import build_np_models, build_approx_gauss_models, sample_gaussian

import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import pylab as pl

def compute_vis_np(model, X, x0):
    kernel, Kab, Da, Dainv, Ra, Kterma, D_ter, R_ter, Xa, Xpa = model
    
    num_act = len(Da)
    betas = [ lu_solve(Da[a], kernel(Xa[a], x0)) for a in xrange(num_act) ]
    
    # drop values corresponding to terminal states
    vals = [ kernel(X, Xp).dot(b[:Xp.shape[0]]) for b, Xp in izip(betas, Xpa) ]

    return vals
    
def compute_vis_explicit( model, X, x0):
    Fa, ra, phi = model
    return [ phi(X).dot(F.dot(phi(x0))) for F in Fa]
    
def compute_vis_explicit_compressed( model, X, x0):
    Fa, ra, phi = model
    U,S, V = Fa[0]
    print U.shape, S.shape, V.shape
    return [ phi(X).dot(U.dot(S.dot(Vt.dot(phi(x0))))) for U,S,Vt in Fa]

def generate_data(domain, policy, num_traj):
    samples = []
    actions = domain.discrete_actions
    for i in xrange(num_traj):
        s_t = domain.reset()
        while s_t is not None:
            if np.random.rand(1) < 0.05:
                a = 1#np.random.randint(3)
            else:
                a = 2 if s_t[1] >= 0 else 0
            r_t, s_tp1 = domain.step(actions[a])
            samples.append((s_t, a, r_t, s_tp1))
            s_t = s_tp1
    domain.random_start= False        
    s_t = domain.reset()
    while s_t is not None:
        a = 2 if s_t[1] >= 0 else 0
        r_t, s_tp1 = domain.step(actions[a])
        samples.append((s_t, a, r_t, s_tp1))
        s_t = s_tp1
                
    sample_a = [ list() for i in xrange(3)]
    term_sample_a = [ list() for i in xrange(3)]
    for s_t, a_t, r_t, s_tp1 in samples:
        if s_tp1 is not None:
            sample_a[a_t].append((s_t, r_t, s_tp1))
        else:
            term_sample_a[a_t].append((s_t, r_t))
            
    for i in xrange(3):
        X, R, Xp = zip(*sample_a[i])
        sample_a[i] = (np.array(X), np.zeros_like(np.array(R)), np.array(Xp))
    #         sample_a[i] = (np.array(X), np.array(R), np.array(Xp))
    
        if len(term_sample_a[i]) > 0:
            X, R = zip(*term_sample_a[i])
            term_sample_a[i] = (np.array(X), np.ones_like(np.array(R)))
    #         term_sample_a[i] = (np.array(X), np.array(R))
        else:
            term_sample_a[i] = (np.array([[]]), np.array([]))
         
        
    term_rew_samples = (sample_a[0][0], np.zeros_like(sample_a[0][1]))
    return sample_a, term_sample_a, term_rew_samples, samples


def fourier_features(X, w):
    if X.ndim == 1:
        X = X.reshape((1,-1))
    features = np.hstack((X, np.ones((X.shape[0], 1)))).dot(w)        
    features = np.hstack((np.sin(features), np.cos(features))) / np.sqrt(w.shape[1])
    return features.squeeze()

def create_Xpa(trans_samples):
    return zip(*trans_samples)[-1]


def plot_samples(S, Sp):
    lines = [ [x0, x1] for x0,x1 in zip(S, Sp) if x1 is not None]
    
    sample_color = [(0,0,0,0.6)]*len(lines)
    
#     print traj, len(traj)
    lc = mc.LineCollection(lines, color = sample_color, linewidths=2)
    plt.gca().add_collection(lc)

#############################################################################
#############################################################################


domain = MountainCar(random_start = True, max_episode = 1000)
s_range = domain.state_range
policy = PumpingPolicy()
width = np.array([0.1, 0.1])

num_gauss = 10000
scale = ((s_range[1] - s_range[0]) * width)
w = sample_gaussian(s_range[0].shape[0], num_gauss, scale)   
phi = lambda X: fourier_features(X, w)

def kernel(X, Y):
    if X.ndim == 1:
        X = X.reshape((1,-1))
        
    if Y.ndim == 1:
        Y = Y.reshape((1,-1))
    scale = ((s_range[1] - s_range[0]) * width)[None,:,None]
        

    # first compute the difference between states
    diff = X[:,:,None] - Y.T[None,:,:]
    
    # get the squared distance
    dsqr = -((diff/scale)**2).sum(axis=1)
    
    return np.exp(dsqr).squeeze()

#def kernel(X, Y):
#    if X.ndim == 1:
#        X = X.reshape((1,-1))
#        
#    if Y.ndim == 1:
#        Y = Y.reshape((1,-1))
#    return phi(X).dot(phi(Y).T).squeeze()



num_traj = 10

trans_samples, ter_samples, ter_rew_samples, samples = generate_data(domain, policy, num_traj)


lamb = 0.2
np_models = build_np_models(kernel, 
                                     trans_samples, 
                                     ter_samples, 
                                     ter_rew_samples, 
                                     lamb)
                                    
np_models = list(np_models)
np_models.append(create_Xpa(trans_samples))                                     
                                     
                                     
phi_models = build_approx_gauss_models(scale, 
                                  trans_samples, 
                                  ter_samples,
                                  phi = phi,
                                  lamb = lamb,
                                  k = 100)
                                  
num_points = 100
ref_point = np.array([-0.3, 0.03])

x = np.linspace(domain.state_range[0][0], domain.state_range[1][0], num_points)
y = np.linspace(domain.state_range[0][1], domain.state_range[1][1], num_points)
X, Y = np.meshgrid(x, y)
vals = compute_vis_np(np_models, np.hstack((X.reshape(-1,1), Y.reshape(-1,1))), ref_point)

f = plt.figure()

i = 1
for a in xrange(3):
    plt.subplot(2, 3, i)
    i += 1
    c = plt.pcolormesh(X, Y, vals[a].reshape((num_points, -1)))
    Xa, R, Xpa = zip(*trans_samples)
    plot_samples(Xa[a], Xpa[a])
    plt.plot([ref_point[0]], [ref_point[1]], 'ko')
    plt.title('np, actions ' + str(a))
    f.colorbar(c)
    

vals = compute_vis_explicit_compressed(phi_models, np.hstack((X.reshape(-1,1), Y.reshape(-1,1))), ref_point)
for a in xrange(3):
    plt.subplot(2, 3, i)
    i += 1
    c = plt.pcolormesh(X, Y, vals[a].reshape((num_points, -1)))
    Xa, R, Xpa = zip(*trans_samples)
    plot_samples(Xa[a], Xpa[a])
    plt.plot([ref_point[0]], [ref_point[1]], 'ko')
    plt.title('phis, actions ' + str(a))
    f.colorbar(c)
#plt.tight_layout()
plt.show()