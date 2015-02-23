from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.theanotools import Theano_RBF_Projector
from rltools.valuefn import TDOF, TDCOF

import numpy as np

import matplotlib.pyplot as plt

import pickle
import time
import sys
time.time()

def episode_data(domain, policy):
    r_t, s_t = domain.reset()
    while s_t is None:
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

def generate_matrices(states, rew, phi):
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

def compute_Theta(valuefn):
    U,S,V = valuefn.matrices[0]
    A,B = valuefn.buffer[0]
    return U.dot(np.diag(S).dot(V.T)) + A.dot(B.T)


def compute_true_theta(domain, policy, n_episodes, gamma, phi):
    states, rew = generate_data(domain, policy, n_episodes)
    X_t, X_tp1, b = generate_matrices(states, rew, phi)
    A = X_t.T.dot(X_t-gamma*X_tp1)
    b = X_t.T.dot(X_t)
    return np.linalg.lstsq(A, b)[0]

domain = MountainCar(random_start=True, max_episode=10000)
policy = PumpingPolicy()

s_range = domain.state_range

xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 20, True),
                     np.linspace(s_range[0][1], s_range[1][1], 20, True))
c = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
w = (s_range[1] - s_range[0])*0.09
phi = Theano_RBF_Projector(c, w)



################ TD PARAMETERS ################
alpha = 0.1
alpha_R = 0.1
lamb = 0.6
gamma = 0.99
n_actions = 1
rank = [10,20,40,80,160,320]
replacing_trace = False
################################################
# 
# tdcof = TDCOF(phi, alpha, alpha_R, lamb, gamma, n_actions, rank, replacing_trace)
# tdof = TDOF(phi, alpha, alpha_R, lamb, gamma, n_actions, replacing_trace)
# 
# t = time.time()
# for i in xrange(10000):
#     tdcof.update(np.random.rand(2), 0, -1.0, np.random.rand(2),0)
# print 'tdcof ', time.time()- t
# 
# t = time.time()
# for i in xrange(10000):
#     tdof.update(np.random.rand(2), 0, -1.0, np.random.rand(2),0)
# print 'tdof ', time.time()- t
# sys.exit()

true_theta = compute_true_theta(domain, policy, 4000, gamma, phi)


num_trials = 5
num_episodes = 3000
screenshot_interval = 20
all_states, all_rew = [],[]
tdcof_score = { k:[] for k in rank}
tdof_score = []
for i in xrange(num_trials):
    print 'starting trial: ' + str(i)
    # GENERATE TRIAL DATA
    states, rew = generate_data(domain, policy, num_episodes)
    all_states += states
    all_rew += rew
    
    # SIMULATE TDCOF, TDOF 
    theta_tdcof = { k:[] for k in rank}
    theta_tdof = []
    index = []
    
    tdcof = {k:TDCOF(phi, alpha, alpha_R, lamb, gamma, n_actions, k, replacing_trace) for k in rank}
    tdof = TDOF(phi, alpha, alpha_R, lamb, gamma, n_actions, replacing_trace)
    
    for j, (traj, traj_r) in enumerate(zip(states, rew)):
        s_t = traj[0]
        for k in rank:
            tdcof[k].update(None, None, None, None, None)
        tdof.update(None, None,None, None, None)
        for s_tp1, r_t in zip(traj[1:], traj_r):
            for k in rank:
                tdcof[k].update(s_t, 0, r_t, s_tp1, 0)
            tdof.update(s_t, 0, r_t, s_tp1, 0)
            s_t = s_tp1
        for k in rank:
            tdcof[k].update(s_t, 0, r_t, None, 0)
        tdof.update(s_t, 0, r_t, None, 0)
        if (j%screenshot_interval) == screenshot_interval-1:
            for k in rank:
                theta_tdcof[k].append(np.linalg.norm(compute_Theta(tdcof[k]) - true_theta))
            theta_tdof.append(np.linalg.norm(tdof.matrices[0] - true_theta))
            index.append(j)
            
    for k in rank:
        tdcof_score[k].append(theta_tdcof[k])
    tdof_score.append(theta_tdof)
    
with open('exp_res-5.data', 'wb') as f:
    pickle.dump( (index, tdcof_score, tdof_score, true_theta,
                  (alpha, alpha_R, lamb, gamma, n_actions, rank, replacing_trace)),
                f)

plt.figure()
plt.plot(index, np.mean(tdcof_score[40], axis=0), label='TDCOF')
plt.plot(index, np.mean(tdof_score, axis=0), label='TDOF')
plt.legend()
plt.show()