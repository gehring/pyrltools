from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.valuefn import TDCOF, LinearTD
from rltools.theanotools import Theano_RBF_Projector
from rltools.agent import TabularActionSarsa
from rltools.policy import Egreedy

import matplotlib.pyplot as plt

import numpy as np

domain = MountainCar(random_start=True, max_episode=10000)
policy = PumpingPolicy()
s_range = domain.state_range

xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10, True),
                     np.linspace(s_range[0][1], s_range[1][1], 10, True))
c = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
w = (s_range[1] - s_range[0])*0.09
phi = Theano_RBF_Projector(c, w)


alpha = 0.1
alpha_R = 0.1
lamb = 0.3
gamma = 0.99
n_actions = 3
rank = 30
replacing_trace = False
epsilon = 0.01

valuefn = TDCOF(phi, alpha, alpha_R, lamb, gamma, n_actions, rank, replacing_trace)
valuefn = LinearTD(3, phi, alpha, lamb, gamma, replacing_trace)

policy = Egreedy(np.arange(3), valuefn, epsilon)

agent = TabularActionSarsa(domain.discrete_actions, policy, valuefn)

num = 50
xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], num, True),
                     np.linspace(s_range[0][1], s_range[1][1], num, True))
grid = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
points = phi(grid)

num_episodes = 1000
for i in xrange(num_episodes):
    r_t, s_t = domain.reset()
    agent.reset()
#     valuefn.update(None, None, None, None, None)
    count =0
    while s_t is not None:
        r_t, s_t = domain.step(agent.step(r_t, s_t))
#         valuefn.update(s_t, 0, r_t, s_tp1, 0)
#         s_t = s_tp1
        count += 1
    print count
    agent.step(r_t, s_t)
    if (i%200) == 199:
        plt.figure()
#         plt.pcolormesh(xx, yy, points.dot(valuefn.get_values()).reshape((num, -1)))
        plt.pcolormesh(xx, yy, (points.dot(valuefn.theta.T)).max(axis=1).reshape((num, -1)))
        plt.colorbar()
#         plt.figure()
#         U = valuefn.matrices[0][0]
#         for i in xrange(12):
#             plt.subplot(4,3,i+1)
#             plt.title('U' + str(i))
#             val = points.dot(U[:,i])
#             plt.pcolormesh(xx.reshape((num,-1)), yy.reshape((num,-1)), val.reshape((num,-1)))
#         plt.matshow(valuefn.matrices[0][0])
#         plt.matshow(np.diag(valuefn.matrices[0][1]))
#         plt.matshow(valuefn.matrices[0][2])
#         plt.matshow(valuefn.buffer[0][0])
#         plt.matshow(valuefn.buffer[0][1])
        plt.show()
    