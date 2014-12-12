from rltools.theanotools import Theano_RBF_Projector
from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.representation import TileCodingDense

from itertools import product

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn import linear_model

import numpy as np
from numpy import meshgrid
from numpy.lib.stride_tricks import as_strided

from scipy.optimize import minimize

def run_episode_from(s_t, domain, policy, gamma):
    rewards, _ = domain.reset()
    domain.state[:] = s_t
    i = 1
    while s_t is not None:
        r_t, s_t = domain.step(policy(s_t))
        rewards += r_t*gamma**i
        i+= 1
    return rewards

def episode_data(domain, policy):
    r_t, s_t = domain.reset()
    states = []
    rew = []
    while s_t is not None:
        states.append(s_t)
        a_t = policy(s_t)
        r_t, s_t = domain.step(a_t)
        rew.append(r_t)
    return np.array(states), np.array(rew)

def generate_data(domain, policy, n_episodes):
    data = [ episode_data(domain, policy) for i in xrange(n_episodes)]
    states, rew = zip(*data)
    return states, rew


def build_lsq(states, rew, phi):
    X_t = []
    X_tp1 = []
    b = []
    for s, r in zip(states, rew):
        X_t.append(s)
        if s.shape[0] > 1:
            X_tp1.append(np.vstack((phi(s[1:]), np.zeros((1,phi.size)))))
        else:
            X_tp1.append(np.zeros((1,phi.size)))
        r = np.array(r)
        b.append(r)
    X_t = np.vstack(X_t)
    X_t = phi(X_t)
    X_tp1 = np.vstack(X_tp1)
    return X_t.T.dot(X_t - X_tp1), X_t.T.dot(np.hstack(b))

def build_MSPBE_minimize(states, rew, phi):
    X_t = []
    X_tp1 = []
    b = []
    for s, r in zip(states, rew):
        X_t.append(s)
        if s.shape[0] > 1:
            X_tp1.append(np.vstack((phi(s[1:]), np.zeros((1,phi.size)))))
        else:
            X_tp1.append(np.zeros((1,phi.size)))
        r = np.array(r)
        b.append(r)
    X_t = np.vstack(X_t)
    X_t = phi(X_t)
    X_tp1 = np.vstack(X_tp1)
    dX = X_t - X_tp1
    R = np.hstack(b)
    
    B = X_t.T.dot(X_t)+ np.identity(X_t.shape[1])
    print np.linalg.matrix_rank(B)
    Binv = np.linalg.pinv(B)
    
    A = dX.T.dot(X_t).dot(Binv).dot(X_t.T.dot(dX))
    C = 2*R.T.dot(X_t).dot(Binv.dot(X_t.T.dot(dX)))
    
    
    
    fn = lambda theta : theta.T.dot(A.dot(theta)) - C.dot(theta)
    
    return fn

domain = MountainCar(False, 1000)
s_range = domain.state_range
policy = PumpingPolicy()
centers = [ [x,y] for x,y in product(np.linspace(s_range[0][0], s_range[1][0], 15, True),
                                     np.linspace(s_range[0][1], s_range[1][1], 15, True))]
# centers *= 2
centers = np.array(centers)
widths = (s_range[1]-s_range[0])*0.15
# centers += (np.random.rand(*centers.shape) - 0.5)*widths



phi = Theano_RBF_Projector(centers, widths, normalized = False)

phit = TileCodingDense([np.arange(2)],
                 [10],
                 [10],
                 hashing=None,
                 state_range = s_range,
                 bias_term=True)

class temp(object):
    def __init__(self, proj):
        self.proj = proj
        self.size = proj.size

    def __call__(self, x):
        if x.ndim == 1:
            return phit(x)
        if x.shape[0] == 1:
            return phit(x[0,:])
        if x.shape[0] == 0:
            return np.zeros(0)
        else:
            return  np.vstack((phit(s) for s in x))
# phi = temp(phit)

xx, yy = meshgrid(np.linspace(s_range[0][0], s_range[1][0], 50, True),
                      np.linspace(s_range[0][1], s_range[1][1], 50, True))
    
points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))

true_val = np.array([ run_episode_from(s, domain, policy, gamma)])

grid = phi(points)

def get_score(num_data, true_val, grid):
    print 'generating data...'
    domain.random_start = True
    states, rew = generate_data(domain, policy, num_data)
    print 'processing and solving...'
    
    
    alpha = 0.0001
    
    # solve with LSQ (i.e., LSTD)
    X,b = build_lsq(states, rew, phi)
    rig = linear_model.Ridge(alpha=alpha, fit_intercept = False)
    clf2 = rig.fit(X, b)
    
    fn = build_MSPBE_minimize(states, rew, phi)
    res = minimize(fn, np.zeros(X.shape[1]))
    theta = res.x
    
    
    val3 = grid.dot(theta) #clf3.predict(grid) #
    val2 = clf2.predict(grid) #grid.dot(theta2)
