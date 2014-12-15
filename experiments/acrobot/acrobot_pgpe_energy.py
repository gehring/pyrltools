from rltools.acrobot import Acrobot
from rltools.policy import evaluate_policy, PGPE

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as stats


def mean_confidence_interval(data, confidence=0.95):
    a = data
    n = a.shape[0]
    m, se = np.mean(a, axis=0), stats.sem(a, axis=0)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, h

start_sampler = lambda : np.array([(np.random.rand()-0.5)*np.pi/4, 0, 0, 0])

domain = Acrobot(random_start = True,
                max_episode = 500,
                m1 = 1,
                m2 = 1,
                l1 = 1,
                l2 = 2,
                g = 9.81,
                b1 = 0.1,
                b2 = 0.1,
                start_sampler = start_sampler)

domain.dt[-1] = 0.01
thres = np.pi/2
domain.goal_range = [np.array([np.pi - thres, -thres, -thres, -thres]),
                  np.array([np.pi + thres, thres, thres, thres]),]

policy = domain.get_swingup_policy()

evaluator = lambda p, k: evaluate_policy(domain, p, k)




def generate_policy(param):
    policy.set_param(param)
    return policy

x0 = np.array([10,10,1, 1000])
sigma0 = np.array([0.1, 0.1, 0.1, 10])

max_iterations = 30
evaluation_iter = 2
final_eval_iter = 2
alpha_mu = 0.1
alpha_sigma = 0.1
alpha_basline = 0.1

num_trials = 4

results = [ PGPE(x0,
                 evaluator,
                 generate_policy,
                 sigma0,
                 max_iterations,
                 evaluation_iter,
                 final_eval_iter,
                 alpha_mu,
                 alpha_sigma,
                 alpha_basline) for _ in xrange(num_trials)]

print zip(*results)[0]
scores = np.array(zip(*results)[1])
print scores[:,-1]
y, yerr = mean_confidence_interval(scores, confidence = 0.95)
plt.errorbar(np.arange(y.shape[0]), y, yerr )
plt.show()