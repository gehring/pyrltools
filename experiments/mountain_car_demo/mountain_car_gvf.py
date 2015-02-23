from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.theanotools import Theano_Tiling

from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np

import matplotlib.pyplot as plt

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

def generate_matrices_sparse(states, rew, phi):
    X_t = []
    X_tp1 = []
    b = []
    for s, r in zip(states, rew):
        X_t.append(s)
        if s.shape[0] > 1:
            B = convert_to_sparse(phi(s[1:]), (s.shape[0]-1, phi.size))
            X_tp1.append(sp.vstack((B, csr_matrix((1, phi.size)))))
        else:
            X_tp1.append(csr_matrix((1, phi.size)))
        r = np.array(r)
        b.append(r)
    X_t = np.vstack(X_t)
    X_t = convert_to_sparse(phi(X_t), (X_t.shape[0], phi.size))
    X_tp1 = sp.vstack(X_tp1)

    return X_t, X_tp1, np.hstack(b)

def generate_matrices(states, rew, phi):
    X_t = []
    X_tp1 = []
    b = []
    for s, r in zip(states, rew):
        X_t.append(s)
        if s.shape[0] > 1:
            X_tp1.append(np.vstack((phi(s[1:]), np.zeros((1, phi.size)))))
        else:
            X_tp1.append(np.zeros((1, phi.size)))
        r = np.array(r)
        b.append(r)
    X_t = np.vstack(X_t)
    X_t = phi(X_t)
    X_tp1 = np.vstack(X_tp1)

    return X_t, X_tp1, np.hstack(b)

def convert_to_sparse(X, dim):
    n = X.shape[1]
    k = X.shape[0]
    indices = X.reshape(-1)
    ptr = np.arange(k+1)*n
    return csr_matrix((np.ones(X.size),
                      indices,
                      ptr),
                      shape = dim)

def get_theta(states, rew, phi):
    X_t, X_tp1, r = generate_matrices_sparse(states, rew, phi)
    A = X_t.T.dot(X_t-X_tp1)
    b = X_t.T.dot(X_t)


    U, s, V = sp.linalg.svds(A, k = sp.linalg.rank(A)*(7.0/8))
    Ainv = V.T.dot((1.0/s)[:,None]*U.T)
    theta = np.array(np.dot(Ainv, b.todense()))
    return theta

class gvf(object):
    def __init__(self, theta, phi):
        self.size = theta.shape[1]
        self.theta = theta
        self.phi = phi

    def __call__(self, states):
        X = convert_to_sparse(self.phi(states), (states.shape[0], self.phi.size))
        return (X.dot(self.theta))

domain = MountainCar(random_start=True, max_episode=3000)
s_range = domain.state_range
a_range = domain.action_range

phi = Theano_Tiling(input_indicies = [np.arange(2)],
                    ntiles = [10],
                    ntilings = [10],
                    hashing = None,
                    state_range = s_range,
                    bias_term = True)

print phi.size

pump_policy = PumpingPolicy()
def choose_action(s):
    if np.random.rand()<0.5:
        return pump_policy(s)
    else:
        return np.random.rand()*(a_range[1] - a_range[0]) + a_range[0]

policy = choose_action #lambda s: np.random.choice(4)#, p= [0.4,0.2,0.2,0.2])

print 'generating data...'
states, rew = generate_data(domain, policy, 4000)

print 'solving...'
X_t, X_tp1, r = generate_matrices_sparse(states, rew, phi)

A = X_t.T.dot(X_t-X_tp1)
b = X_t.T.dot(X_t)

br = X_t.T.dot(r)


U, s, V = sp.linalg.svds(A, k=800)
Ainv = V.T.dot((1.0/s)[:,None]*U.T)
print Ainv.shape, b.todense().shape
theta = np.array(np.dot(Ainv, b.todense()))
print theta.shape

if sp.issparse(theta):
    print 'sparse'
    U, s, V = sp.linalg.svds(theta, k=80)
else:
    U, s, V = np.linalg.svd(theta)
theta_v = sp.linalg.lsmr(A,br)[0]
# theta = np.array([sp.linalg.lsmr(A, np.array(bs.todense()).reshape(-1))[0] for bs in b.T]).T
# U, s, V = np.linalg.svd(theta)



print 'evaluating and plotting...'
num = 40
xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], num),
                     np.linspace(s_range[0][1], s_range[1][1], num))
points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
grid = convert_to_sparse(phi(points), (points.shape[0], phi.size))

print U.shape, V.shape, V[1,:].shape
print type(V[1,:])
print grid.shape
fig = plt.figure(figsize=(12,8))
plt.plot(s, linewidth = 3)
fig.tight_layout()
plt.savefig('svd-mountain-tiling.pdf')


X_t, X_tp1, r = generate_matrices_sparse(states, rew, phi)

X_t = X_t.tocsr()
X_tp1 = X_tp1.tocsr()

n_vec = 12
n_row = 4

plt.figure()
for i in xrange(n_vec):
    plt.subplot(n_row,n_vec/n_row + 0 if n_vec % n_row == 0 else 1,i+1)
    plt.title('U' + str(i))
    val = grid.dot(U[:,i])
    plt.pcolormesh(xx.reshape((num,-1)), yy.reshape((num,-1)), val.reshape((num,-1)))

plt.figure()
for i in xrange(n_vec):
    plt.subplot(n_row,n_vec/n_row + 0 if n_vec % n_row == 0 else 1,i+1)
    plt.title('V' + str(i))
#     
#     A = X_t.T.dot(X_t-X_tp1)
#     b = X_t.T.dot(X_t.dot(V[i,:].T))
#     lsq = sp.linalg.lsmr(A, b)[0]
#     
    val = grid.dot(V[i,:].T)
#     val = grid.dot(lsq)
    plt.pcolormesh(xx.reshape((num,-1)), yy.reshape((num,-1)), val.reshape((num,-1)))

plt.figure()
plt.subplot(1,2,1)
plt.title('Full Value Fn')
val = grid.dot(theta_v)
true_val = val
plt.pcolormesh(xx.reshape((num,-1)), yy.reshape((num,-1)), val.reshape((num,-1)))

plt.subplot(1,2,2)
plt.title('Approx Value Fn')
val = grid.dot(U[:,:n_vec].dot(np.diag(s[:n_vec]).dot(V[:n_vec,:])).dot(-np.ones(theta.shape[1])))
plt.pcolormesh(xx.reshape((num,-1)), yy.reshape((num,-1)), val.reshape((num,-1)))

print 'generating data...'
states, rew = generate_data(domain, policy, 500)

print 'solving...'

thetaphi = gvf(U[:,:n_vec].dot(np.diag(s[:n_vec]).dot(V[:n_vec,:])), phi)
# thetaphi = gvf(V[:8,:].T, phi)
# thetaphi = gvf(U[:,:20], phi)
thetaX_t, thetaX_tp1, r = generate_matrices(states, rew, thetaphi)
thetagrid = thetaphi(points)

# thetaX_t = thetaX_t.todense()
# thetaX_tp1 = thetaX_tp1.todense()

n = X_t.shape[0]/10
indices = range(1,n+1, int(n/10))
lsq_score = []
theta_score = []
for i in indices:
    A = X_t[:i,:].T.dot(X_t[:i,:]-X_tp1[:i,:])
    b = X_t[:i,:].T.dot(r[:i])
    lsq = sp.linalg.lsmr(A, b)[0]

    lsq_score.append(np.linalg.norm(true_val - grid.dot(lsq))/ np.linalg.norm(true_val))

    thetaA = thetaX_t[:i,:].T.dot(thetaX_t[:i,:]-thetaX_tp1[:i,:])
    thetab = thetaX_t[:i,:].T.dot(r[:i])
    thetalsq = np.linalg.lstsq(thetaA, thetab)[0]
    
    theta_score.append(np.linalg.norm(true_val - thetagrid.dot(thetalsq))/ np.linalg.norm(true_val))


plt.figure()
plt.subplot(2,2,1)
plt.title('lstd')
plt.plot(indices, lsq_score)
plt.ylim(0.0, 0.8)

plt.subplot(2,2,2)
plt.title('gvf lstd')
plt.plot(indices, theta_score)
plt.ylim(0.0, 0.8)

plt.subplot(2,2,3)
plt.pcolormesh(xx.reshape((num,-1)), yy.reshape((num,-1)), grid.dot(lsq).reshape((num,-1)))

plt.subplot(2,2,4)
plt.pcolormesh(xx.reshape((num,-1)), yy.reshape((num,-1)), thetagrid.dot(thetalsq).reshape((num,-1)))


plt.show()

