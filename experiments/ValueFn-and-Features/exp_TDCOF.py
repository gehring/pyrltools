from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.theanotools import Theano_RBF_Projector
from rltools.valuefn import TDOF, InvTDCOF, TDCOF

import numpy as np

import matplotlib.pyplot as plt

import pickle
import time
import sys
# import winsound
time.time()

class TDCOF(object):
    def __init__(self,
                 projector,
                 alpha,
                 alpha_R,
                 lamb,
                 gamma,
                 n_actions,
                 rank = None,
                 replacing_trace=True):
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_R = alpha_R
        self.phi = projector

        self.Theta = np.eye(self.phi.size)
        self.R = np.zeros(self.phi.size)
        self.count = 0

    def __call__(self, state, action = None):
        phi_s = self.phi(state)

        if action is None:
            v = np.zeros(self.count.shape[0])
            for i, ((U,S,V), (A,B)) in enumerate(zip(self.matrices, self.buffer)):
                Uphi = phi_s.T.dot(U)
                phiA = phi_s.T.dot(A)
                v[i] = self.R.dot(V.dot(np.diag(S).dot(Uphi.T))) + self.R.dot(B.dot(phiA.T))
        else:
            U,S,V = self.matrices[action]
            A, B = self.buffer[action]
            Uphi = phi_s.T.dot(U)
            phiA = phi_s.T.dot(A)
            v = self.R.dot(V.dot(np.diag(S).dot(Uphi.T))) + self.R.dot(B.dot(phiA.T))
        return v


    def call_no_act(self, state):
        phi_s = self.phi(state)
        U,S,V = self.matrices[0]
        A, B = self.buffer[0]
        Uphi = phi_s.T.dot(U)
        phiA = phi_s.T.dot(A)
        v = self.R.dot(V.dot(np.diag(S).dot(Uphi.T))) + self.R.dot(B.dot(phiA.T))
        return v

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t is None:
            return

        phi_t = self.phi(s_t)
        self.count +=1
        # eval V(s_t)
        v_t = (phi_t.T.dot(self.Theta)).T

        if s_tp1 is not None:
            # eval v_tp1
            phi_tp1 = self.phi(s_tp1)
            v_tp1 = (phi_tp1.T.dot(self.Theta)).T
            delta = phi_t + self.gamma*v_tp1 - v_t
            norm_coef = (phi_t - self.gamma * phi_tp1).dot(self.Theta).dot(phi_t)
        else:
            delta = phi_t - v_t
            norm_coef = (phi_t).dot(self.Theta).dot(phi_t)

        # update svd of the occupancy function
        delta =delta.reshape((1,-1))
        #(1/(1 + norm_coef))
        self.Theta += (0.5/(1 + norm_coef))*( self.Theta.dot(phi_t)).reshape((-1,1)).dot(delta)
        R = self.R
        self.R = R +  self.alpha_R * (r - np.squeeze(phi_t.T.dot(self.R))) * phi_t
#         self.R = np.squeeze(np.array(self.R))

    def update_svd(self, matrices, ab_buffer, initialized):
        U,S,V = matrices
        A, B = ab_buffer
        if initialized:
            Q_a, R_a = np.linalg.qr(A - U.dot(U.T.dot(A)), mode='reduced')
            Q_b, R_b = np.linalg.qr(B - V.dot(V.T.dot(B)), mode='reduced')

            Ap = np.vstack((U.T.dot(A), R_a))
            Bp = np.vstack((V.T.dot(B), R_b))
            K = np.diag(np.hstack((S, np.zeros(R_a.shape[0])))) + Ap.dot(Bp.T)
            Up, Sp, Vp = np.linalg.svd(K, full_matrices = False)

            U = np.hstack((U, Q_a)).dot(Up)
            V = np.hstack((V, Q_b)).dot(Vp.T)

        else:
            Q_a, R_a = np.linalg.qr(A, mode='reduced')
            Q_b, R_b = np.linalg.qr(B, mode='reduced')
            Up, Sp, Vp = np.linalg.svd(R_a.dot(R_b.T), full_matrices = False)

            U = Q_a.dot(Up)
            V = Q_b.dot(Vp.T)
        S = Sp[:self.rank]
        U = U[:,0:self.rank]
        V = V[:,0:self.rank]
        return (U, S, V)

    def correct_orthogonality(self):
        U, S, V = self.matrices
        Vq, Vr = np.linalg.qr(V)
        Uq, Ur = np.linalg.qr(U)
        tU, tS, tV = np.linalg.svd(Ur.dot(np.diag(S)).dot(Vr.T), full_matrices = False)
        V = Vq.dot(tV)
        U = Uq.dot(tU)
        S = tS
        self.matrices = (U,S,V)
#         if self.use_U_only:
#             self.R = self.R.dot(tV)
    def get_values(self,R = None):
        if R is None:
            R = self.R
        v = np.zeros((self.phi.size, 1))
        v[:,0] = self.Theta.dot(R)
        return v

    @property
    def theta(self):
        return self.get_values().T



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
#     A,B = valuefn.buffer[0]
    return U.dot(np.diag(S).dot(V.T))# + A.dot(B.T)


def compute_true_theta(domain, policy, n_episodes, gamma, phi):
    states, rew = generate_data(domain, policy, n_episodes)
    X_t, X_tp1, b = generate_matrices(states, rew, phi)
    A = X_t.T.dot(X_t-gamma*X_tp1)
    b = X_t.T.dot(X_t)
    return np.linalg.lstsq(A, b)[0]

def get_R(domain, policy, n_episodes, phi):
    states, rew = generate_data(domain, policy, n_episodes)
    X_t, X_tp1, b = generate_matrices(states, rew, phi)
    return np.linalg.lstsq(X_t.T.dot(X_t), X_t.T.dot(b))[0]
    
domain = MountainCar(random_start=True, max_episode=10000)
policy = PumpingPolicy()

s_range = domain.state_range

xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10, True),
                     np.linspace(s_range[0][1], s_range[1][1], 10, True))
c = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
w = (s_range[1] - s_range[0])*0.09
phi = Theano_RBF_Projector(c, w)



################ TD PARAMETERS ################
alpha = 0.1
alpha_R = 0.1
lamb = 0.6
gamma = 0.99
n_actions = 1
rank =[240]# [ 60, 120, 240]
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

#R = get_R(domain, policy, 6000, phi)
#with open('rew.data', 'wb') as f:
#     pickle.dump( R, f)
# sys.exit()
with open('rew.data', 'rb') as f:
    R = pickle.load(f)


true_theta = compute_true_theta(domain, policy, 2000, gamma, phi)
print true_theta.shape
# for i in xrange(2):
#     winsound.Beep(450,250)
#     time.sleep(0.25)

# U,S,V = np.linalg.svd(true_theta, full_matrices=False)
# 
# r_true = 401
# true_svd = (U[:,:r_true], S[:r_true], V.T[:,:r_true])

#TDCOF = SRTD

num_trials = 1
num_episodes = 500
screenshot_interval = 20
all_states, all_rew = [],[]
tdcof_score = { k:[] for k in rank}
tdof_score = []

tdcof_score_t = { k:[] for k in rank}
tdof_score_t = []
for i in xrange(num_trials):
    print 'starting trial: ' + str(i)
    # GENERATE TRIAL DATA
    states, rew = generate_data(domain, policy, num_episodes)
    all_states += states
    all_rew += rew

    # SIMULATE TDCOF, TDOF
    theta_tdcof = { k:[] for k in rank}
    theta_tdof = []
    
    theta_tdcof_t = { k:[] for k in rank}
    theta_tdof_t = []
    
    index = []

    tdcof = {k:TDCOF(phi, alpha, alpha_R, lamb, gamma, n_actions, k, replacing_trace) for k in rank}
    tdof = TDOF(phi, alpha, alpha_R, lamb, gamma, n_actions, replacing_trace)

    for j, (traj, traj_r) in enumerate(zip(states, rew)):
        if len(traj)> 1:
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
    #                 print compute_Theta(tdcof[k]).shape, R.shape, true_theta.shape
                    theta_tdcof[k].append(np.linalg.norm((tdcof[k].Theta - true_theta).dot(R)))
                theta_tdof.append(np.linalg.norm((tdof.matrices[0] - true_theta).dot(R)))
                
                for k in rank:
                    theta_tdcof_t[k].append(np.linalg.norm((tdcof[k].Theta - true_theta)))
                theta_tdof_t.append(np.linalg.norm((tdof.matrices[0] - true_theta)))
                
                index.append(j)

    for k in rank:
        tdcof_score[k].append(theta_tdcof[k])
    tdof_score.append(theta_tdof)
    
    for k in rank:
        tdcof_score_t[k].append(theta_tdcof_t[k])
    tdof_score_t.append(theta_tdof_t)

with open('exp_res-inv-no_true.data', 'wb') as f:
    pickle.dump( (index, tdcof_score, tdof_score, true_theta,
                  (alpha, alpha_R, lamb, gamma, n_actions, rank, replacing_trace)),
                f)
    
approx_thetas = {k:tdcof[k].get_values(R) for k in rank}
td_thetas = tdof.get_values(R) 
 
with open('exp_res-inv-theta_no_true.data', 'wb') as f:
     pickle.dump((approx_thetas, td_thetas), f)



# for i in xrange(4):
#     winsound.Beep(450,250)
#     time.sleep(0.25)

plt.figure()
plt.plot(index, np.mean(tdcof_score[rank[0]], axis=0), label='TDCOF')
plt.plot(index, np.mean(tdof_score, axis=0), label='TDOF')
plt.legend()

plt.show()