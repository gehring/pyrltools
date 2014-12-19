from rltools.GridWorld import GridWorld, obstacle_condition, boundary_condition

from itertools import product, izip

from sklearn import linear_model

import numpy as np

import matplotlib.pyplot as plt

class IdenProj(object):
    def __init__(self, shape):
        self.size = np.prod(shape)
        self.shape = shape
    def __call__(self, state):
        if state.ndim > 1:
            phi = np.zeros((state.shape[0],self.size))
            index = np.ravel_multi_index(state.T.astype('int'), self.shape)
            phi[np.arange(state.shape[0]), index] = 1
        else:
            phi = np.zeros(self.size)
            index = np.ravel_multi_index(state.astype('int'), self.shape)
            phi[ index] = 1
        return phi

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

def generate_matrices(states, rew):
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
    
    return X_t, X_tp1, np.hstack(b)
s_range = [np.zeros(2, dtype='int'), np.ones(2, dtype='int')*9]
boundary = boundary_condition(s_range)

walls = list(product([5], [0,2,3,4,5,6,7,9])) \
        + list(product([0,2,3,4], [5])) \
        + list(product([5,6,8,9], [4]))
obstacles = obstacle_condition(set(walls))

islegal_check = lambda s: boundary(s) and obstacles(s)

reward_fn = lambda s_t, a_t, s_tp1: -1 if np.any(s_tp1 > 1) else 10
termination_fn = lambda x: np.random.rand()<0.001

domain = GridWorld(reward_fn, 
                   islegal_check, 
                   termination_fn, 
                   s_range, 
                   random_start = True,
                   max_episode = 1000)

phi = IdenProj((10,10))

def choose_action(s):
#     if s[0] >= 5 and s[1] >= 5:
#         p = [0.4,0.1,0.4,0.1]
#     if s[0] >= 5 and s[1] < 5:
#         p = [0.1,0.1,0.7,0.1]
#     if s[0] < 5 and s[1] >= 5:
#         p = [0.7,0.1,0.1,0.1]
#     if s[0] < 5 and s[1] < 5:
#         p = [0.4,0.1,0.4,0.1]
    p = np.ones(4)/4.0
    return np.random.choice(4, p=p)
        
policy = choose_action #lambda s: np.random.choice(4)#, p= [0.4,0.2,0.2,0.2])

print 'generating data...'
states, rew = generate_data(domain, policy, 500)

print 'solving...'
X_t, X_tp1, r = generate_matrices(states, rew)

A = X_t.T.dot(X_t-X_tp1)
# b = X_t.T.dot([1 if s.nonzero()[0] == np.ravel_multi_index((7,3), (10,10)) else 0 for s in X_t])
# b = X_t.T.dot(r)
b = X_t.T.dot(X_t)

alpha = 0.0
# theta = linear_model.ridge_regression(A,b, alpha)
theta = np.linalg.lstsq(A, b)[0]

print 'evaluating and plotting...'
xx, yy = np.meshgrid(np.arange(10), np.arange(10))
points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
grid = phi(points)



U, s, V = np.linalg.svd(theta)

plt.figure()
plt.plot(s)

plt.figure()
for i in xrange(5):
    plt.title('Us')
    plt.subplot(1,5,i+1)
    val = grid.dot(U[:,i])
    plt.imshow(val.reshape((10,-1)), interpolation = 'none')

plt.figure()
for i in xrange(5):
    plt.title('Vs')
    plt.subplot(1,5,i+1)
    val = grid.dot(V[i,:])
    plt.imshow(val.reshape((10,-1)), interpolation = 'none')


# R = np.zeros(theta.shape[0])
# R[11] = 1
# R[88] = 1
# val = grid.dot(theta.dot(R))
# # val = grid.dot(theta)
# plt.figure()
# plt.imshow(val.reshape((10,-1)), interpolation = 'none')

plt.show()



