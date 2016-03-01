import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


from rltools.MountainCar import MountainCar
from rltools.theanotools import Theano_RBF_Projector

matplotlib.rcParams.update({'font.size': 14})

with open('exp_res-inv-no_true.data', 'rb') as f:
    (index, tdcof_score, tdof_score, theta,
                  (alpha, alpha_R, lamb, gamma, n_actions, rank, replacing_trace)) =pickle.load(f)


s = (12,8)
fig = plt.figure(figsize=s)
for k in rank:
    plt.plot(index, np.mean(tdcof_score[k], axis=0), label='ALSR, k=' + str(k), linewidth=3)
plt.plot(index, np.mean(tdof_score, axis=0), label='SR', linewidth=3)
plt.xlabel('Number of episodes')
plt.ylabel('$\parallel \Theta - \hat{\Theta} \parallel_F$')
plt.legend()
fig.tight_layout()
plt.savefig('all.pdf')


fig = plt.figure(figsize=s)
plt.plot(rank, [np.mean(tdcof_score[k], axis=0)[-1] for k in rank ], color='r', label='ALSR', linewidth=3)
plt.plot([0,30], [np.mean(tdof_score, axis=0)[-1]]*2, 'k--', label='SR', linewidth=4)
plt.xlabel('rank')
plt.ylabel('$\parallel \Theta - \hat{\Theta} \parallel_F$')
plt.legend()
fig.tight_layout()
plt.savefig('rank.pdf')

U,S,V = np.linalg.svd(theta)
fig = plt.figure(figsize = s)
plt.plot(S, linewidth = 3)
fig.tight_layout()
plt.savefig('svd-mountain-rbf.pdf')

cs = np.cumsum(S)/np.sum(S)
fig = plt.figure(figsize = s)
plt.plot(cs, linewidth = 3)
fig.tight_layout()
plt.savefig('cum-svd-mountain-rbf.pdf')

# print (cs <0.99).nonzero()[-1].shape
# print S[(cs <0.99).nonzero()[-1][-1]]



domain = MountainCar(random_start=True, max_episode=10000)

s_range = domain.state_range

xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10, True),
                     np.linspace(s_range[0][1], s_range[1][1], 10, True))
c = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
w = (s_range[1] - s_range[0])*0.09
phi = Theano_RBF_Projector(c, w)


with open('exp_res-inv-theta_no_true.data', 'rb') as f:
    (approx_thetas, td_thetas) = pickle.load(f)

num = 40
xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 40, True),
                     np.linspace(s_range[0][1], s_range[1][1], 40, True))
grid = phi(np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))))
for r in rank:
    plt.figure(figsize=s)
    plt.pcolormesh(xx, yy, grid.dot(approx_thetas[r]).reshape((num, -1)))
    plt.title(str(r))
    plt.colorbar()

plt.figure(figsize=s)
plt.pcolormesh(xx, yy, grid.dot(td_thetas).reshape((num, -1)))
plt.title('sr')
plt.colorbar()


plt.show()