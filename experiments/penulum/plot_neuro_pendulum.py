import matplotlib.pyplot as plt
import pickle
import numpy as np
from rltools.SwingPendulum import SwingPendulum

def find_best(score, key_index):
    best = {}
    argbest = {}
    for k in score.keys():
        v = k[key_index]
        if v in best:
            if not np.isnan(score[k]):
                if score[k] < best[v]:
                    best[v] = score[k]
                    argbest[v] = k
        else:
            if not np.isnan(score[k]):
                best[v] = score[k]
                argbest[v] = k
    return best, argbest

def filter_dict(score, test):
    new_score = {}
    for k in score.keys():
        if test(k):
            new_score[k] = score[k]
    return new_score

filename = '/media/cgehri/data/experiment_data/pendulum/test-0-complete-data.data'
with open(filename, 'rb') as f:
    results, avg_rew, true_val, params = pickle.load(f)
    rates, alphas, alpha_mus, etas = params
print avg_rew

scores = {}
values = {}
n = np.ones_like(results[0][0])
n /= np.linalg.norm(n)
for p, param in results:
    cr = param['rate']
    a = true_val[cr]
#     r = (a - p) - ((a-p).dot(n)) * n
    x = np.mean(p-a)
    r = a-p+x
    key = (param['rate'], param['alpha'], param['alpha_mu'], param['eta'])
    scores[key] = np.sqrt(np.mean(r**2))
    values[key] = p
    
    
plt.figure()
for e in etas:
    score_eta = filter_dict(scores, lambda k: k[3] == e)
    best_rate,_ = find_best(score_eta, 0)
    p = [a for a in rates if a in best_rate]
    plot_score = [ best_rate[cr] for cr in p]
    plt.plot(p, plot_score, label = 'eta = '+str(e))
    
plt.xlabel('control rate')
plt.ylabel('error')
plt.legend()

plt.figure()
for e in etas:
    score_eta = filter_dict(scores, lambda k: k[3] == e)
    best_rate,_ = find_best(score_eta, 1)
    p = [a for a in alphas if a in best_rate]
    plot_score = [ best_rate[cr] for cr in p]
    plt.plot(p, plot_score, label = 'eta = '+str(e))
    
plt.xlabel('alpha')
plt.ylabel('error')
plt.xscale('log', nonposy='clip')
plt.legend()


for e in rates:
    plt.figure()
    score_eta = filter_dict(scores, lambda k: k[0] == e)
    best_rate,_ = find_best(score_eta, 3)
    p = [a for a in etas if a in best_rate]
    plot_score = [ best_rate[cr] for cr in p]
    plt.plot(p, plot_score, label = 'rate = '+str(e))
        
    plt.xlabel('eta')
    plt.ylabel('error')
    plt.legend()





plt.figure()
domain = SwingPendulum(random_start=True)
s_range = domain.state_range
nsamples = 40        
xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], nsamples),
                     np.linspace(s_range[0][0], s_range[1][0], nsamples))
points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
num = len(true_val)
row = int(np.floor(np.sqrt(num)))
col = int(np.ceil(np.sqrt(num)))
for i, r in enumerate(true_val.keys()):
    plt.subplot(row,col,i+1)
    plt.pcolormesh(xx, yy, (true_val[r] - avg_rew[r]).reshape(xx.shape))
    plt.title(str(r))
    plt.colorbar()
    
plt.figure()
num = len(etas)
row = int(np.floor(np.sqrt(num)))
col = int(np.ceil(np.sqrt(num)))

score_rate = filter_dict(scores, lambda k: k[0] == min(rates))
best_rate, arg_best = find_best(score_rate, 3)
for i, e in enumerate(etas):
    k = arg_best[e]
    plt.subplot(row,col,i+1)
    plt.pcolormesh(xx, yy, values[k].reshape(xx.shape))
    plt.title(str(e))
    plt.colorbar()

# for r, param in results:
#     if param['rate'] == 0.01:
#         plt.figure()
#         plt.pcolormesh(xx, yy, r.reshape(xx.shape))
#         plt.title(str(param))
#         plt.colorbar()
plt.show()

