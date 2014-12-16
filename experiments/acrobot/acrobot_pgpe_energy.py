from rltools.acrobot import Acrobot
from rltools.policy import evaluate_policy, PGPE

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as stats

from IPython.parallel import Client

from itertools import repeat, izip

import pyprind
import pickle


def mean_confidence_interval(data, confidence=0.95):
    a = data
    n = a.shape[0]
    m, se = np.mean(a, axis=0), stats.sem(a, axis=0)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, h


class acrobot_evaluator(object):
    def __init__(self, domain):
        self.domain=domain
        
    def __call__(self, p, k):
        return evaluate_policy(self.domain, p, k)
    
class policy_generator(object):
    def __init__(self, policy):
        self.policy = policy
        
    def __call__(self, param):
        self.policy.set_param(param)
        return self.policy
    
def start_sampler():
    return np.array([(np.random.rand()-0.5)*np.pi/4, 0, 0, 0])

def run_trial(p):
    return PGPE(*p)

domain = Acrobot(random_start = True,
                max_episode = 1000,
                m1 = 1,
                m2 = 1,
                l1 = 1,
                l2 = 2,
                g = 9.81,
                b1 = 0.1,
                b2 = 0.1,
                start_sampler = start_sampler)

domain.dt[-1] = 0.01
thres = np.pi/4
domain.goal_range = [np.array([np.pi - thres, -thres, -thres, -thres]),
                  np.array([np.pi + thres, thres, thres, thres]),]

policy = domain.get_swingup_policy()

evaluator = acrobot_evaluator(domain)






x0 = np.array([10.0,10.0,1.0, 600.0])
sigma0 = np.array([0.1, 0.1, 0.1, 10])*10

max_iterations = 100
evaluation_iter = 10
final_eval_iter = 10
alpha_mu = np.array([0.2, 0.2, 0.2, 0.2])
alpha_sigma = 0.1
alpha_baseline = 0.1

num_trials = 36

client = Client()
client[:].execute('from rltools.acrobot import Acrobot', block=True)
client[:].execute('from rltools.policy import PGPE, evaluate_policy', block=True)
client[:].execute('import numpy as np', block=True)
    
client[:]['acrobot_evaluator'] = acrobot_evaluator
client[:]['policy_generator'] = policy_generator
client[:]['start_sampler'] = start_sampler

lbview =  client.load_balanced_view()
lbview.block = False
lbview.retries = True

# compute total number of runs
X0 = [ x0.copy() for i in xrange(num_trials)]
S0 = [ sigma0.copy() for i in xrange(num_trials)]
domains = [ domain.copy() for i in xrange(num_trials)]
evals = [ acrobot_evaluator(d) for d in domains]
generators = [ policy_generator(d.get_swingup_policy()) for d in domains]

results = lbview.map( run_trial, izip(X0, 
                                               evals,
                                               generators,
                                               S0,
                                               repeat(max_iterations),
                                               repeat(evaluation_iter),
                                               repeat(final_eval_iter),
                                               repeat(alpha_mu),
                                               repeat(alpha_sigma),
                                               repeat(alpha_baseline)),
                     ordered = False,
                    block = False)

bar = pyprind.ProgBar(num_trials)

for s in results:
    bar.update()

try:
    with open('pgpe_results2.data', 'wb') as f:
        pickle.dump(results, f)
except Exception as e:
    pass


print zip(*results)[0]
scores = np.array(zip(*results)[1])
print scores[:,-1]
y, yerr = mean_confidence_interval(scores, confidence = 0.95)
plt.errorbar(np.arange(y.shape[0]), y, yerr )
plt.show()