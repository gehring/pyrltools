
from rltools.acrobot import Acrobot, get_trajectories, get_qs_from_traj,\
    compute_acrobot_from_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

domain = Acrobot(random_start = False,
                 m1 = 1,
                 m2 = 1,
                 l1 = 1,
                 l2=2,
                 b1=0.0,
                 b2=0.0)
domain.start_state[0] = 0.01
domain.dt[-1] = 0.01
domain.action_range = [np.array([-10]), np.array([10])]

print 'generating trajectories...'
controller = lambda q: np.random.rand()*20-10
states, torques = get_trajectories(domain, 1, 2000, controller = controller)
q, qd, qdd, y = get_qs_from_traj(states, torques, domain.dt[-1])
# qdd = np.vstack((domain.state_dot(np.hstack((q[i,:], qd[i,:])), 0, y[i])[2:] for i in xrange(q.shape[0])))
a_list = []

indices = range(200,2001,200)
for i in indices:
    id_domain = compute_acrobot_from_data(q[:i,:], qd[:i,:], qdd[:i,:], y[:i], method = 'dynamics', random_start = False)
    a_list.append(id_domain.a)

m1 = domain.m1
m2 = domain.m2
I1 = domain.Ic1
I2 = domain.Ic2
l1 = domain.l1
l2 = domain.l2
lc1 = domain.lc1
lc2 = domain.lc2
g = domain.g
b1, b2 = domain.b1, domain.b2


a = np.array([m1*lc1**2 + m2*l1**2 + m2*lc2**2+ I1 + I2,
              m2*l1*lc2,
              m2*lc2**2 + I2,
              (m1*lc1 + m2*l1)*g,
              m2*lc2*g,
              b1,
              b2])

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

lw = 4.0
for i,(a_values, a_true) in enumerate(zip(zip(*a_list), a)):
    f = plt.figure()
    plt.plot(indices, [a_true]*(len(a_values)), 'r', label='True Value', linewidth=lw)
    plt.plot(indices, a_values, 'b', label='Fitted Value', linewidth=lw)
    plt.plot(indices, a_values, 'ko', linewidth=lw)
    plt.legend(loc=2)
    plt.xlabel('Number of Samples')
    plt.ylabel('Value')
#     plt.savefig('a'+str(i)+'.pdf', bbox_inches='tight')
plt.show()


