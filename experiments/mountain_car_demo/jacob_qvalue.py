from rltools.theanotools import Theano_RBF_stateaction
from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.policy import Egreedy

import matplotlib.pyplot as plt

import numpy as np

from sklearn import linear_model

from itertools import product

domain = MountainCar(False, 5000)
s_range = domain.state_range
a_range = domain.action_range
actions = np.linspace(a_range[0], a_range[1], 3, True).reshape((-1,1))# np.array(domain.discrete_actions).reshape((-1,1))
policy = PumpingPolicy()

class Valuefn(object):
    def __init__(self, actions, theta, phi):
        self.theta = theta
        self.actions = actions
        self.phi = phi
        
    def __call__(self, state, action = None):
        if action == None:
            action = self.actions
        return self.phi(state, action).dot(self.theta)
    
def generate_data(domain, samples, a_range):
    s_t, s_tp1, r_t, a_t = [],[],[],[]
    for s in samples:
        domain.state[:] = s
        a = actions[np.random.choice(actions.shape[0])]
        r, sp = domain.step(a)
        s_t.append(s)
        s_tp1.append(sp)
        r_t.append(r)
        a_t.append(a)
    return s_t, a_t, r_t, s_tp1

def build_opt(s_t, a_t, r_t, s_tp1, phi, fn, t):
    X1, X2 = [], []
    b1, b2 = [], []
    
    # filter terminal states
    for s,a,r,sp in zip(s_t, a_t, r_t, s_tp1):
        if sp is None:
            X2.append((s,a))
            b2.append(r)
        else:
            X1.append((s,a,sp - s))
            b1.append(-r)
    S, A, dS = zip(*X1)
    S, A, dS = np.array(S), np.array(A), np.array(dS)
    
    w = np.sqrt( np.exp(fn(S,A)/t) / np.array( [np.sum(np.exp(fn(s)/t)) for s in S]))
    X1 = phi.getdphids(S, A, dS) * w[:,None]
    b1 = np.array(b1) * w
    
#     X1 = phi.getdphids(S, A, dS)
#     b1 = np.array(b1)

    S, A = zip(*X2)
    S, A = np.array(S), np.array(A)
    w = np.sqrt( np.exp(fn(S,A)/t) / np.array( [np.sum(np.exp(fn(s)/t)) for s in S]))
    X2 = phi(S,A) * w[:,None]
    b2 = np.array(b2) * w
#     X2 = phi(S,A) 
#     b2 = np.array(b2)
    return np.vstack((X1,X2)), np.hstack((b1,b2))

def build_opt_filter(s_t, a_t, r_t, s_tp1, phi, valuefn):
    X1, X2 = [], []
    b1, b2 = [], []
    
    # filter terminal states
    for s,a,r,sp in zip(s_t, a_t, r_t, s_tp1):
        if actions[np.argmax(valuefn(s))] == a:
            if sp is None:
                X2.append((s,a))
                b2.append(r)
            else:
                X1.append((s,a,sp - s))
                b1.append(-r)
#         else:
#             print actions[np.argmax(valuefn(s))], a
    S, A, dS = zip(*X1)
    S, A, dS = np.array(S), np.array(A), np.array(dS)
    
    X1 = phi.getdphids(S, A, dS)
    b1 = np.array(b1)

    S, A = zip(*X2)
    S, A = np.array(S), np.array(A)

    X2 = phi(S,A) 
    b2 = np.array(b2)
    return np.vstack((X1,X2)), np.hstack((b1,b2))
    
    

centers = [ c for c in product(np.linspace(s_range[0][0], s_range[1][0], 15, True),
                                     np.linspace(s_range[0][1], s_range[1][1], 15, True),
                                     np.linspace(a_range[0][0], a_range[1][0], 3, True))]
centers *= 2
centers = np.array(centers)
widths = (s_range[1]-s_range[0])*0.15
widths = np.hstack((widths, (a_range[1] - a_range[0])*0.1))
centers += (np.random.rand(*centers.shape) - 0.5)*widths

phi = Theano_RBF_stateaction(centers, widths, normalized = True)


num = 30
xx, yy = np.meshgrid(*([np.linspace(0, 1, num, True)]*2))

samples = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))) *(s_range[1]-s_range[0]) + s_range[0]
samples = np.vstack([samples]*2)
print 'generating data...'
data = generate_data(domain, samples, domain.action_range)

print 'processing data...'
fn = Valuefn(actions, np.random.rand(phi.size)*0.01, phi)
X,b = build_opt(*data, phi=phi, fn=fn, t=1)
# X,b = build_opt_filter(*data, phi=phi, valuefn=fn)
alpha = 0.001

print 'optimizing...'
theta = linear_model.ridge_regression(X, b, alpha)


for i in range(200):
    fn = Valuefn(actions, theta, phi)
    X,b = build_opt(*data, phi=phi, fn=fn, t=0.1)
#     X,b = build_opt_filter(*data, phi=phi, valuefn=fn)
    alpha = 0.001
     
    theta = linear_model.ridge_regression(X, b, alpha)

# theta = np.linalg.solve((X.T.dot(X) + alpha*np.eye(X.shape[1])), X.T.dot(b))

print 'evaluating and plotting...'
xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 50, True),
                  np.linspace(s_range[0][1], s_range[1][1], 50, True))

points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))

n_actions = 3
sa = [ p for p in product(points, np.linspace(a_range[0], a_range[1],  n_actions,True))]
S,A = zip(*sa)
S,A = np.array(S), np.array(A).reshape((-1,1))
grid = phi(S,A)
values = grid.dot(theta)
val = np.max(values.reshape((-1, n_actions)), axis=1)
pol = np.argmax(values.reshape((-1, n_actions)), axis=1)

plt.figure()
plt.contourf(xx, yy, val.reshape((xx.shape[0],-1)))
plt.colorbar()
plt.title('SF with Jac')

plt.figure()
plt.contourf(xx, yy, pol.reshape((xx.shape[0],-1)))
plt.colorbar()
plt.title('SF with Jac, policy')

plt.show()
