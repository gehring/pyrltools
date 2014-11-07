import numpy as np
from rltools.RDG import RDG
from rltools.POMaze import createMazeFromLines
from rltools.policy import Egreedy
from rltools.agent import POSarsa
from rltools.valuefn import linearValueFn, LinearTD
from rltools.representation import IdentityProj
from sets import Set
from itertools import repeat, chain
import pickle
import matplotlib.pyplot as plt

size = (8,8)
goal = Set([(5,5)])
walls = [((1,0), (1,1)),
         ((2,0), (3,0)),
         ((4,0), (4,1)),
         ((5,0), (5,1)),
         ((0,1), (1,1)),
         ((1,1), (1,2)),
         ((2,1), (2,2)),
         ((2,1), (3,1)),
         ((3,1), (4,1)),
         ((5,1), (5,2)),
         ((6,1), (7,1)),
         ((1,2), (1,3)),
         ((3,2), (3,3)),
         ((3,2), (4,2)),
         ((4,2), (5,2)),
         ((5,2), (6,2)),
         ((6,2), (7,2)),
         ((0,2), (1,3)), #((0,3), (1,3))
         ((1,3), (2,3)),
         ((2,3), (3,3)),
         ((3,3), (3,4)),
         ((4,3), (4,4)),
         ((4,3), (5,3)),
         ((5,3), (6,3)),
         ((6,3), (7,3)),
         ((5,4), (6,4)),
         ((6,4), (6,5)),
         ((6,4), (7,4)),
         ((0,5), (1,5)),
         ((1,5), (2,5)),
         ((4,5), (4,6)),
         ((6,5), (6,6)),
         ((0,6), (1,6)),
         ((2,6), (1,6)),
         ((4,6), (3,7)), #((3,6), (3,7))
         ((4,6), (4,7)),
         ((2,6), (3,6)),
         ((4,6), (5,6)),
         ((5,6), (6,6)),
         ((0,7), (1,7)),
         ((1,7), (2,7)),
         ((2,5), (3,5)),
         ((3,5), (4,5))]
maze = createMazeFromLines(walls, goal, size)

num_graphs = 5000
num_nodes = 100
rdg = RDG(num_graphs, num_nodes, num_obs=16*4)

# policy = lambda x: np.random.randint(0, 4)
#
# data = []
# for i in xrange(500):
#     r, s_t = maze.reset()
#     a_t = policy(s_t) if s_t != None else None
#     traj = []
#     while s_t != None:
#         s_tm1 = s_t
#         a_tm1 = a_t
#         r, s_t = maze.step(a_t)
#         a_t = policy(s_t) if s_t != None else None
#         traj.append((s_tm1, a_tm1, r, s_t, a_t))
#     data.append(traj)
#
# with open('traj.data', 'wb') as f:
#     pickle.dump(data, f)
with open('traj.data', 'rb') as f:
    data = pickle.load(f)

print 'data loaded!'

# gamma = 0.99
# lamb = 0.2
#
# valuefn = LinearTD(1, IdentityProj(rdg.size), 0.001, lamb, gamma, True)
#
# phi = rdg
# # A = np.zeros((phi.size, phi.size))
# # b = np.zeros(phi.size)
# print 'fitting theta...'
# for i, t in enumerate(chain(*repeat(data, 2))):
#     rdg.reset()
#     rdg.update(np.ravel_multi_index(t[0][:2], (16,4)))
#     x_t = rdg.getState()
#     z =x_t
#     time = 0
#     print'trajectory '+str(i) + ' is being processed...'
#     for sarsa in t:
#         if sarsa[3] == None:
#             x_tp1 = np.zeros(0, dtype='uint')
#         else:
#             rdg.update(np.ravel_multi_index(sarsa[3:], (16,4)))
#             x_tp1 = rdg.getState()
#         r = sarsa[2]
#         valuefn.update(x_t, 0, r, x_tp1, 0)
#
# #         A += np.outer(z, x_t - gamma*x_tp1)
# #         b += z * r
# #         z = gamma*lamb * z + x_tp1
#         x_t = x_tp1
#         time += 1
#     valuefn.update(x_t, 0, r, x_tp1, 0)
#     if np.any(np.isnan(valuefn.theta)):
#         print "NaN found! ABORT!"
#         break
#     if i%10 == 0:
#         val = linearValueFn(valuefn.theta[0,:], valuefn.phi)
#         with open('value.data', 'wb') as f:
#             pickle.dump((rdg, val), f)
#
# # theta = np.linalg.lstsq(A, b)[0]
# # val = linearValueFn(theta, phi)
# val = linearValueFn(valuefn.theta[0,:], valuefn.phi)
# print 'Value function computed!'
# with open('value.data', 'wb') as f:
#     pickle.dump((rdg, val), f)

with open('value.data', 'rb') as f:
    rdg, val = pickle.load(f)

values = []
rdg.reset()
for s,a,r,s_tp1, a_tp1 in data[0]:
    rdg.update(np.ravel_multi_index((s,a), (16,4)))
    x_t = rdg.getState()
    values.append(np.sum(val.theta[x_t]))

values2 = []
rdg.reset()
for s,a,r,s_tp1, a_tp1 in data[1]:
    rdg.update(np.ravel_multi_index((s,a), (16,4)))
    x_t = rdg.getState()
    values2.append(np.sum(val.theta[x_t]))

plt.plot(values)
plt.plot(values2)
plt.show()





