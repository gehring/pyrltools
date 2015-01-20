from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.theanotools import Theano_RBF_Projector, Theano_Tiling
from rltools.mathtools import iSVD
from rltools.representation import TileCodingDense

import matplotlib.pyplot as plt
import numpy as np


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

def build_lsq(states, rew, phi, gamma):
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
    return X_t.T.dot(X_t - gamma*X_tp1), X_t.T.dot(np.hstack(b))

def build_Theta(states, rew, phi, gamma):
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
    return X_t.T.dot(X_t - gamma*X_tp1), X_t.T.dot(X_t)


def update_svd(isvd, s_t, r_t, s_tp1, phi, alpha, gamma):
    U, S, V = isvd.get_decomp()
    p_t = phi(s_t)

    if s_tp1 is not None:
        td = p_t + gamma* V.T.dot(np.diag(S).dot(U.T.dot(phi(s_tp1)))) \
                        - V.T.dot(np.diag(S).dot(U.T.dot(p_t)))
    else:
        td = p_t - V.T.dot(np.diag(S).dot(U.T.dot(p_t)))

    isvd.update(p_t, alpha*td)

def update_td(theta, s_t, r_t, s_tp1, phi, alpha, gamma):
    p_t = phi(s_t)

    if s_tp1 is not None:
        td = r_t + gamma*theta.dot(phi(s_tp1)) \
                        - theta.dot(p_t)
    else:
        td = r_t - theta.dot(p_t)

    theta += alpha*td*p_t

def update_r_td(U, S, V, R, s_t, r_t, s_tp1, phi, alpha, gamma):
    p_t = phi(s_t)

    if s_tp1 is not None:
        td = r_t + gamma* R.dot(V.T.dot(np.diag(S).dot(U.T.dot(phi(s_tp1))))) \
                        - R.dot(V.T.dot(np.diag(S).dot(U.T.dot(p_t))))
    else:
        td = r_t - R.dot(V.T.dot(np.diag(S).dot(U.T.dot(p_t))))
    R += alpha*td*p_t

def update_r_regression(R, s_t, r_t, s_tp1, phi, alpha):
    p_t = phi(s_t)
    R += alpha*(r_t - R.dot(p_t)) * p_t

def generate_transition(states, rewards):
    for i in xrange(states.shape[0]):
        if i < states.shape[0]-1:
            yield states[i], rewards[i], states[i+1]
        else:
            yield states[i], rewards[i], None

domain = MountainCar(random_start = True, max_episode = 1000)
policy = PumpingPolicy()

s_range = domain.state_range

xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10, True),
                     np.linspace(s_range[0][1], s_range[1][1], 10, True))
centers = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
widths = (s_range[1]-s_range[0])*0.2
phi = Theano_RBF_Projector(centers, widths, bias_term=True, normalized = True)
phi = TileCodingDense([np.arange(2)], [5], [11], None, s_range, True)

num = 3000
print 'Generating data...'
allstates, rew = generate_data(domain, policy, n_episodes= num)

isvd = iSVD(max_rank=100, shape=(phi.size, phi.size), init = False)
R_reg = np.zeros(phi.size)
R_td = np.zeros_like(R_reg)
theta_td = np.zeros_like(R_reg)

alpha = 0.1
gamma = 0.99

# empirical truth
print 'Solving empirical truth...'
A, b = build_lsq(allstates, rew, phi, gamma)
theta_true = np.linalg.lstsq(A, b)[0]


# scores
r_reg_score = []
r_td_score = []
td_score = []


xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 40, True),
                     np.linspace(s_range[0][1], s_range[1][1], 40, True))
points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
grid = phi(points)

V_true = grid.dot(theta_true)

run_for = min(50, num)
print 'Running all incremental algorithms:'
for i, (states, rewards) in enumerate(zip(allstates, rew)[:run_for]):
    for s_t, r_t, s_tp1 in generate_transition(states, rewards):
        update_svd(isvd, s_t, r_t, s_tp1, phi, alpha, gamma)
        update_r_regression(R_reg, s_t, r_t, s_tp1, phi, alpha)
        U, S, V = isvd.get_decomp()
        update_r_td(U, S, V, R_td, s_t, r_t, s_tp1, phi, alpha, gamma)
        update_td(theta_td, s_t, r_t, s_tp1, phi, alpha, gamma)
    if i % (run_for/10) == 0:
        print i
        V_td = U.dot(np.diag(S).dot(V.dot(R_td)))
        V_reg =  U.dot(np.diag(S).dot(V.dot(R_reg)))
#         V_reg =  U.dot(np.diag(S).dot(V.dot(-np.ones(phi.size)/3.0)))
        r_reg_score.append(np.linalg.norm(grid.dot(V_reg) - V_true))
        r_td_score.append(np.linalg.norm(grid.dot(V_td) - V_true))
        td_score.append(np.linalg.norm(grid.dot(theta_td) - V_true))

print 'Plotting...'
plt.Figure()
plt.plot(range(1,len(r_reg_score)+1), r_reg_score, label='R reg')
plt.plot(range(1,len(r_reg_score)+1), td_score, label='td')
plt.legend()

# plt.subplot(3,1,1)
# plt.plot(range(1,len(r_reg_score)+1), r_reg_score)
# plt.ylim(0,1200)
# plt.title('R reg')
#
# plt.subplot(3,1,2)
# plt.plot(range(1,len(r_reg_score)+1), r_td_score)
# plt.ylim(0,1200)
# plt.title('R td')
#
# plt.subplot(3,1,3)
# plt.plot(range(1,len(r_reg_score)+1), td_score)
# plt.ylim(0,1200)
# plt.title('TD')

# plt.figure()
# A, B = build_Theta(allstates, rew, phi, gamma)
# Ut, St, Vt =np.linalg.svd(A)
# THETA = Vt.T.dot(np.diag(1.0/St).dot(Ut.T)).dot(B)
# Ut, St, Vt =np.linalg.svd(THETA)
# for i in xrange(12):
#     plt.subplot(3,4,i+1)
#     plt.pcolormesh(xx.reshape((40,40)), yy.reshape((40,40)), (grid.dot(Ut[:,i])).reshape((40,40)))
#
# plt.figure()
# for i in xrange(12):
#     plt.subplot(3,4,i+1)
#     plt.pcolormesh(xx.reshape((40,40)), yy.reshape((40,40)), (grid.dot(U[:,i])).reshape((40,40)))

plt.figure()
plt.subplot(2,2,1)
plt.pcolormesh(xx.reshape((40,40)), yy.reshape((40,40)), grid.dot(V_reg).reshape((40,40)))
plt.subplot(2,2,2)
plt.pcolormesh(xx.reshape((40,40)), yy.reshape((40,40)), grid.dot(V_td).reshape((40,40)))
plt.subplot(2,2,3)
plt.pcolormesh(xx.reshape((40,40)), yy.reshape((40,40)), grid.dot(theta_td).reshape((40,40)))
plt.subplot(2,2,4)
plt.pcolormesh(xx.reshape((40,40)), yy.reshape((40,40)), V_true.reshape((40,40)))

plt.show()
