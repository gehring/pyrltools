from rltools.theanotools import Theano_NRBF_Projector,Theano_RBF_Projector
from rltools.MountainCar import MountainCar, PumpingPolicy

from itertools import product

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn import linear_model

import numpy as np
from numpy import meshgrid

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

def build_opt(states, rew, phi):
    X = []
    b = []
    for s, r in zip(states, rew):
        ds = s[1:,:]-s[:-1,:]
        X.append(phi.getdphids(s[:-1,:], ds))
        X.append(phi(s[-1]))
        r = np.array(r)
        r[-1] *= -1
        b.append( -r)

    return np.vstack(X), np.hstack(b)

def build_lsq(states, rew, phi):
    X_t = []
    X_tp1 = []
    b = []
    for s, r in zip(states, rew):
        X_t.append(s)
        X_tp1.append(np.vstack((phi(s[1:]), np.zeros((1,phi.size)))))
        r = np.array(r)
        b.append(r)
    X_t = phi(np.vstack(X_t))
    X_tp1 = np.vstack(X_tp1)
    return X_t.T.dot(X_t - X_tp1), X_t.T.dot(np.hstack(b))

domain = MountainCar(True, 1000)
s_range = domain.state_range
policy = PumpingPolicy()
centers = [ [x,y] for x,y in product(np.linspace(s_range[0][0], s_range[1][0], 15, True),
                                     np.linspace(s_range[0][1], s_range[1][1], 15, True))]
widths = (s_range[1]-s_range[0])*0.1

phi = Theano_NRBF_Projector(np.array(centers), widths)

print 'generating data...'
states, rew = generate_data(domain, policy, 100)

print 'processing and solving...'

rcond = 0.01

# solve SF with jacobian
X, b = build_opt(states, rew, phi)
alpha = 0.001
# theta = np.linalg.lstsq(X, b, rcond = rcond)[0]
# rig = linear_model.RidgeCV(alphas=[0.001, 0.05, 0.01, 0.1])
rig = linear_model.Ridge(alpha=alpha, fit_intercept = False)
# rig = linear_model.LinearRegression()
clf = rig.fit(X, b)
theta2 = clf.coef_
theta = linear_model.ridge_regression(X, b, alpha)
# theta = np.linalg.solve((X.T.dot(X) + alpha*np.eye(X.shape[1])), X.T.dot(b))

# solve with LSQ (i.e., LSTD)
# X,b = build_lsq(states, rew, phi)
# # theta2 = np.linalg.lstsq(X, b, rcond = rcond)[0]
# #
# rig = linear_model.RidgeCV(alphas=[0.001, 0.05, 0.01, 0.1])
# clf = rig.fit(X, b)
# theta2 = clf.coef_


xx, yy = meshgrid(np.linspace(s_range[0][0], s_range[1][0], 50, True),
                  np.linspace(s_range[0][1], s_range[1][1], 50, True))

points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))

grid = phi(points)
val = grid.dot(theta)
val2 = grid.dot(theta2)
plt.figure()
plt.contourf(xx, yy, val.reshape((xx.shape[0],-1)))
for s in (states[:50] if len(states)>50 else states):
    x, y = zip(*s)
    plt.plot(x, y, 'k', linewidth=2.0, alpha = 0.3)
plt.colorbar()
plt.title('SF with Jac')

plt.figure()
plt.contourf(xx, yy, val2.reshape((xx.shape[0],-1)))
for s in (states[:50] if len(states)>50 else states):
    x, y = zip(*s)
    plt.plot(x, y, 'k', linewidth=2.0, alpha = 0.3)
plt.colorbar()
plt.title('LSTD')

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(xx, yy,val.reshape((xx.shape[0],-1)), rstride=1, cstride=1,
#                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
# # for s,r in (zip(states[:10], rew[:10]) if len(states)>10 else zip(states,rew)):
# #     x, y = zip(*s)
# #     z = -np.cumsum(r)
# #     z -= z.max()
# #     plt.plot(x, y, z, 'k', linewidth=2.0, alpha = 0.3)
# plt.title('SF')
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(xx, yy, val2.reshape((xx.shape[0],-1)), rstride=1, cstride=1,
#                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
# plt.title('LSTD')
plt.show()
