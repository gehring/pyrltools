import matplotlib.pyplot as plt
import pickle
import numpy as np


def find_best(score, key_index):
    best = {}
    for k in score.keys():
        v = k[key_index]
        if v in best:
            best[v] = min(best[v], score[k])
        else:
            best[v] = score[k]
    return best

def filter_dict(score, test):
    new_score = {}
    for k in score.keys():
        if test(k):
            new_score[k] = score[k]
    return new_score

filename = 'complete-data.data'
with open(filename, 'rb') as f:
    results, avg_rew, true_val = pickle.load(f)
    
print avg_rew
print true_val.keys()


scores = {}
n = np.ones_like(results[0][0])
n /= np.linalg.norm(n)
for p, param in results:
    cr = param['rate']
    a = true_val[cr]
    r = (a - p) - ((a-p).dot(n)) * n
    key = param['rate'], param['alpha'], param['alpha_mu'], param['eta']
    scores[key] = np.linalg.norm(r)
    
    
etas = [0.0, 0.3, 0.6, 0.9]
rates = [0.2, 0.1, 0.05, 0.01]
plt.figure()
for e in etas:
    score_eta = filter_dict(scores, lambda k: k[3] == e)
    best_rate = find_best(score_eta, 0)
    plot_score = [ best_rate[cr] for cr in rates]
    plt.plot(cr, plot_score, label = 'eta = '+str(e))
    
plt.show()