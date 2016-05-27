# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:58:11 2016

@author: cgehri
"""
import matplotlib.pyplot as plt
import numpy as np

from itertools import izip

from matplotlib import collections  as mc

from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.npplanning import sample_gaussian

from rltools.model import SingleCompressModel

def compute_model_predictions(model, phi, X):
    Kab, Da, wa, Vas, Uas = model
    phi_t = phi(X)
    num_act = len(Da)
    
    vals = [ wa[a].dot(np.diag(Da[a]).dot( Kab[a,a].dot(np.diag(Da[a]).dot(Vas[a].T.dot(phi_t.T))))) for a in xrange(num_act) ]    
    return vals
    
def generate_data(domain, policy, num_traj, average = False):
    samples = []
    actions = domain.discrete_actions
    sampled_traj = []
    for i in xrange(num_traj-1):
        traj = []
        s_t = domain.reset()
        while s_t is not None:
            if np.random.rand(1) < 0.05:
                a = 1#np.random.randint(3)
            else:
                a = 2 if s_t[1] > 0 else 0
            r_t, s_tp1 = domain.step(actions[a])
            traj.append((s_t, actions[a], r_t, s_tp1))
            samples.append((s_t, actions[a], r_t, s_tp1))
            s_t = s_tp1
        sampled_traj.append(traj)
        
    domain.random_start= False        
    s_t = domain.reset()
    traj = []
    while s_t is not None:
        if np.random.rand(1) < 0.1:
            a = 1#np.random.randint(3)
        else:
            a = 2 if s_t[1] > 0 else 0
        r_t, s_tp1 = domain.step(actions[a])
        traj.append((s_t, actions[a], r_t, s_tp1))
        samples.append((s_t, actions[a], r_t, s_tp1))
        s_t = s_tp1
    sampled_traj.append(traj)
    
    X_t = []
    A_t = []
    X_tp1 = []
    X_term = []
    A_term = []
    R_t = []
    R_term = []
    for traj in sampled_traj:
        for s_t, a, r_t, s_tp1 in traj:
            if s_tp1 is None:
                X_term.append(s_t)
                R_term.append(r_t)
#                R_term.append(1)
                A_term.append(a)
            else:
                X_t.append(s_t)
                R_t.append(r_t)
#                R_t.append(0)
                A_t.append(a)
                X_tp1.append(s_tp1)
    return (np.array(X_t), np.array(A_t), np.array(R_t), np.array(X_tp1),
            np.array(X_term), np.array(A_term), np.array(R_term), samples)
        
                
                
def fourier_features(X, w):
    if X.ndim == 1:
        X = X.reshape((1,-1))
    features = np.hstack((X, np.ones((X.shape[0], 1)))).dot(w)        
    features = np.hstack((np.sin(features), np.cos(features))) / np.sqrt(w.shape[1])
    return features.squeeze()
    
def get_next_state(domain, state, action):
    domain.reset()
    domain.state = state.copy()
    return domain.step(action)[1]
    
def display_arrows(states, next_states):
    ax = plt.gca()
    for s_t, s_tp1 in izip(states, next_states):
        if s_tp1 is not None:
            ax.arrow(s_t[0], s_t[1], s_tp1[0] -s_t[0], s_tp1[1]- s_t[1], head_length = 0.02, fc = 'k', ec='k', width=0.0001)
            
def plot_samples(S, Sp):
    lines = [ [x0, x1] for x0,x1 in zip(S, Sp) if x1 is not None]
    
    sample_color = [(0,0,1.0,0.8)]*len(lines)
    
#     print traj, len(traj)
    lc = mc.LineCollection(lines, color = sample_color, linewidths=2)
    plt.gca().add_collection(lc)
            
def create_graph(filename, average = False, use_kernel = False, num_basis = 10):

    domain = MountainCar(random_start = True, max_episode = 1000)
    s_range = domain.state_range
    policy = PumpingPolicy()
    width = np.array([0.1, 0.1])

    num_gauss = num_basis
    scale = ((s_range[1] - s_range[0]) * width)
    w = sample_gaussian(s_range[0].shape[0], num_gauss, scale)
    #w = np.random.rand(2, num_gauss)
    #w = w/scale[:,None]
    #w = np.vstack((w, np.zeros((1, num_gauss))))   
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



    num_traj = 1

    X_t, A_t, R_t, X_tp1, X_term, A_term, R_term, samples = generate_data(domain, policy, num_traj, average)

    lamb = 0.2

    phi_t = phi(np.vstack((X_t, X_term)))
    phi_tp1 = np.vstack((phi(X_tp1), np.zeros((X_term.shape[0], phi_t.shape[1]))))
    R_t = np.hstack((R_t, R_term))
    A_t = np.vstack((A_t, A_term))
    
    model =SingleCompressModel(dim = num_gauss*2,
                               action_dim = 1,
                               max_rank = 500,
                               lamb = lamb,
                               X_t = phi_t,
                               A_t = A_t,
                               R_t = R_t,
                               X_tp1 = phi_tp1)
     
    actions = [ np.array([-0.001]),
                          np.array([0]),
                          np.array([0.001])]            
    def single_action_kernel(i):
        #return lambda b: np.exp(-np.sum(((actions[i][None,:]-b)/0.002)**2, axis=1)/(1.0**2))
        return lambda b: (actions[i][None,:] == b).astype('float').squeeze()
        #return lambda b: np.ones(b.shape[0])
    
    action_kernels = [ single_action_kernel(i) for i in xrange(len(actions))]                
           
           
    model.CompressedX_t.update_and_clear_buffer()
    models = model.generate_embedded_model(action_kernels)


                                      
    num_points = 100
    ref_point = np.array([-0.3, 0.05])

    arrow_grid = 10
    x = np.linspace(domain.state_range[0][0], domain.state_range[1][0], arrow_grid)
    y = np.linspace(domain.state_range[0][1], domain.state_range[1][1], arrow_grid)
    X, Y = np.meshgrid(x, y)
    states = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
    next_state=[]
    for a in xrange(3):
        next_state.append([])
        for s in states:
            next_state[a].append(get_next_state(domain, s, domain.discrete_actions[a]))


    x = np.linspace(domain.state_range[0][0], domain.state_range[1][0], num_points)
    y = np.linspace(domain.state_range[0][1], domain.state_range[1][1], num_points)
    X, Y = np.meshgrid(x, y)
    f = plt.figure(figsize=(10,3))
    plt.subplots_adjust(wspace = 0.3)
        
    i = 1

    vals = compute_model_predictions(models, phi, np.hstack((X.reshape(-1,1), Y.reshape(-1,1))))
    for a in xrange(3):
        if a == 1:
            continue
        plt.subplot(1, 2, i)
        i += 1
        c = plt.pcolormesh(X, Y, vals[a].reshape((num_points, -1)), cmap='Oranges')
        display_arrows(states, next_state[a])
        Xa, Xpa = zip(*[ (x_t, x_tp1) for x_t, a_t, r_t, x_tp1 in samples if np.allclose(a_t, actions[a])])
        plot_samples(Xa, Xpa)
        #plt.plot([ref_point[0]], [ref_point[1]], 'ro')
        plt.title('actions ' + str(a))
        f.colorbar(c)

    plt.show()
    #plt.savefig(filename)
    #plt.close()
    
create_graph('exp-fourier-10-avg.png', average = False, use_kernel = False, num_basis  = 5000)