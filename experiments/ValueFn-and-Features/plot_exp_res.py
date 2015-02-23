import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 14})

with open('exp_res-proto-3.data', 'rb') as f:
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
plt.plot([0,400], [np.mean(tdof_score, axis=0)[-1]]*2, 'k--', label='SR', linewidth=4)
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

plt.show()