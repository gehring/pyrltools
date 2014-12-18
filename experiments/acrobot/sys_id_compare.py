
from rltools.acrobot import Acrobot, get_trajectories, get_qs_from_traj,\
    compute_acrobot_from_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import scipy.stats as stats

def mean_confidence_interval(data, confidence=0.95):
    a = data
    n = a.shape[0]
    m, se = np.mean(a, axis=0), stats.sem(a, axis=0)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, h

def get_params(method, indices, q, qd, qdd, y):
    a_list = []
    for i in indices:
        id_domain = compute_acrobot_from_data(q[:i,:],
                                              qd[:i,:],
                                              qdd[:i,:],
                                              y[:i],
                                              method = method,
                                              random_start = False)
        a_list.append(id_domain.a)
    return a_list

def plot_param(indices, means, conf):
#     font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 10}
#
#     matplotlib.rc('font', **font)

    lw = 2.0
    plt.figure(figsize=(30,10))
    for i in xrange(7):
        plt.subplot(2,4,i+1)
        plt.errorbar(indices, means[:,i], conf[:,i], linewidth=lw)
        plt.plot(indices, np.zeros(len(indices)), 'r', linewidth=lw)
        plt.gca().set_title('$x_'+str(i)+'$')
        plt.xlabel('Number of samples')
        plt.ylabel('Absolute relative deviation')
        if i >5:
            plt.ylim(0, 6)
        elif i==5:
            plt.ylim(0,3)
        else:
            plt.ylim(0, 2)

domain = Acrobot(random_start = False,
                 m1 = 1,
                 m2 = 1,
                 l1 = 1,
                 l2=2,
                 b1=0.1,
                 b2=0.1)
domain.start_state[0] = 0.01
domain.dt[-1] = 0.01
domain.action_range = [np.array([-10]), np.array([10])]


controller = lambda q: np.random.rand()*40-20
states, torques = get_trajectories(domain, 1, 1000, controller = controller)
q, qd, qdd, y = get_qs_from_traj(states, torques, domain.dt[-1])

num_samples = 1000
indices = range(num_samples/10,num_samples+1,num_samples/10)

power_param = []
energy_param = []
dynamic_param = []

num_trials = 20

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
print a

# compute all fitted parameters
for i in xrange(num_trials):
    print 'Running trial #'+ str(i+1)+'...'
    states, torques = get_trajectories(domain, 1, num_samples, controller = controller)
    states += np.random.normal(0, 0.05, size = states.shape)
    q, qd, qdd, y = get_qs_from_traj(states, torques, domain.dt[-1])

    power_param.append(get_params('power', indices, q, qd, qdd, y))
    energy_param.append(get_params('energy', indices, q, qd, qdd, y))
    dynamic_param.append(get_params('dynamics', indices, q, qd, qdd, y))




power_param = np.array(power_param)
energy_param = np.array(energy_param)
dynamic_param = np.array(dynamic_param)

power_param -= a[None,None,:]
power_param = np.abs(power_param)/a[None,None,:]
power_mean, power_conf = mean_confidence_interval(power_param, confidence = 0.95)

plot_param(indices, power_mean, power_conf)
plt.savefig('power-noise.pdf', bbox_inches='tight')

energy_param -= a[None,None,:]
energy_param = np.abs(energy_param)/a[None,None,:]
energy_mean, energy_conf = mean_confidence_interval(energy_param, confidence = 0.95)

plot_param(indices, energy_mean, energy_conf)
plt.savefig('energy-noise.pdf', bbox_inches='tight')

dynamic_param -= a[None,None,:]
dynamic_param = np.abs(dynamic_param)/a[None,None,:]
dyn_mean, dyn_conf = mean_confidence_interval(dynamic_param, confidence = 0.95)

plot_param(indices, dyn_mean, dyn_conf)
plt.savefig('dynamics-noise.pdf', bbox_inches='tight')

plt.show()


