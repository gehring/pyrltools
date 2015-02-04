import matplotlib.pyplot as plt
import pickle
import numpy as np
from rltools.SwingPendulum import SwingPendulum

def find_best(score, key_index, compare = None):
    best = {}
    argbest = {}
    if compare is None:
        compare = lambda v0, v1: v0>v1
    for k in score.keys():
        v = k[key_index]
        if v in best:
            if not np.any(np.isnan(score[k])):
                if compare(score[k], best[v]):
                    best[v] = score[k]
                    argbest[v] = k
        else:
            if not np.any(np.isnan(score[k])):
                best[v] = score[k]
                argbest[v] = k
    return best, argbest

def filter_dict(score, test):
    new_score = {}
    for k in score.keys():
        if test(k):
            new_score[k] = score[k]
    return new_score

filename = '/media/cgehri/data/experiment_data/mountaincar/test-200-complete-data.data'
with open(filename, 'rb') as f:
    results, params = pickle.load(f)

alphas, alpha_mus, etas = [params[k] for k in ['alpha', 'alpha_mu', 'eta']]

scores = {}
score_key = ['alpha', 'alpha_mu', 'eta']
for p, param in results:
    key = tuple([param[k] for k in score_key]) 
    if key not in scores:
        scores[key] = []
    
    scores[key].append(p)
    
means = dict(scores)
for k, v in means.iteritems():
    means[k] = np.mean(v, axis=0)

plt.figure()

for e in etas:
    score_eta = filter_dict(means, lambda k: k[2] == e)
    best_rate,_ = find_best(score_eta, 0, lambda v0, v1: v0[-1]>v1[-1])
    p = [a for a in alphas if a in best_rate]
    plot_score = [ best_rate[cr][-1] for cr in p]
    plt.plot(p, plot_score, label = 'eta = '+str(e))
    
plt.xlabel('alpha')
plt.ylabel('avg rew')
plt.xscale('log', nonposy='clip')
plt.legend()

plt.figure()
best_rate,argbest = find_best(means, 2, lambda v0, v1: v0[-1]>v1[-1])
for e in etas:
    plt.plot(np.arange(15), best_rate[e], label = 'param = '+str(argbest[e]))
plt.xlabel('step')
plt.ylabel('avg rew')
plt.legend()

# 
# for e in rates:
#     plt.figure()
#     score_eta = filter_dict(scores, lambda k: k[0] == e)
#     best_rate,_ = find_best(score_eta, 3)
#     p = [a for a in etas if a in best_rate]
#     plot_score = [ best_rate[cr] for cr in p]
#     plt.plot(p, plot_score, label = 'rate = '+str(e))
#         
#     plt.xlabel('eta')
#     plt.ylabel('error')
#     plt.legend()
# 
# 
# 
# 
# 
# plt.figure()
# domain = SwingPendulum(random_start=True)
# s_range = domain.state_range
# nsamples = 40        
# xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], nsamples),
#                      np.linspace(s_range[0][1], s_range[1][1], nsamples))
# points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
# num = len(true_val)
# row = int(np.floor(np.sqrt(num)))
# col = int(np.ceil(np.sqrt(num)))
# for i, r in enumerate(true_val.keys()):
#     plt.subplot(row,col,i+1)
#     plt.pcolormesh(xx, yy, (true_val[r] - avg_rew[r]).reshape(xx.shape))
#     plt.title(str(r))
#     plt.colorbar()
#     
# plt.figure()
# num = len(etas)
# row = int(np.floor(np.sqrt(num)))
# col = int(np.ceil(np.sqrt(num)))
# 
# score_rate = filter_dict(scores, lambda k: k[0] == min(rates))
# best_rate, arg_best = find_best(score_rate, 3)
# for i, e in enumerate(etas):
#     k = arg_best[e]
#     print k
#     plt.subplot(row,col,i+1)
#     plt.pcolormesh(xx, yy, values[k].reshape(xx.shape))
#     plt.title(str(e))
#     plt.colorbar()

# for r, param in results:
#     if param['rate'] == 0.01:
#         plt.figure()
#         plt.pcolormesh(xx, yy, r.reshape(xx.shape))
#         plt.title(str(param))
#         plt.colorbar()
plt.show()

