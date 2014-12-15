import numpy as np
from numpy.random import random_sample
from rltools.valuefn import LSTDlambda

from scipy.stats import multivariate_normal

class Policy(object):
    def __init__(self):
        pass

    def __call__(self, state):
        p = self.getprob(state)
        return self.actions[weighted_values(p)[0]]

class Egreedy(Policy):
    def __init__(self, actions, valuefn, **argk):
        super(Egreedy, self).__init__()
        self.actions = actions
        self.valuefn = valuefn
        self.epsilon = argk.get('epsilon', 0.1)


    def getprob(self, state):
        values = self.valuefn(state)

        # argmax with random tie breaking
        m = np.random.choice(np.argwhere(values == np.amax(values)).flatten(),1)[0]

        values[:] = self.epsilon/len(self.actions)
        values[m] += 1-self.epsilon
        return values


class Egreedy_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        if 'actions' in params:
            return Egreedy(**params)
        else:
            domain = params.get('domain')
            return Egreedy(domain.discrete_actions, **params)

class MixedPolicy(object):
    def __init__(self, p1, p2, policy_ratio):
        self.p1 = p1
        self.p2 = p2
        self.ratio = policy_ratio
    def __call__(self, state):
        if np.random.uniform() < self.ratio:
            return self.p1(state)
        else:
            return self.p2(state)

class MixedPolicy_Factory(object):
    def __init__(self, p1_fact, p2_fact, **argk):
        self.p1_fact = p1_fact
        self.p2_fact = p2_fact
        self.params = argk
    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        return MixedPolicy(self.p1_fact(**params),
                           self.p2_fact(**params),
                           **params)

class SoftMax(Policy):
    def __init__(self, actions, valuefn, **argk):
        super(SoftMax, self).__init__()
        self.actions = actions
        self.value_fn = valuefn
        self.temp = argk.get('temperature', 0.1)

    def getprob(self, state):
        values = self.value_fn(state)
        values /= self.temp
        ev = np.exp(values)
        return ev/np.sum(ev)

class SoftMax_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        domain = params.get('domain')
        return SoftMax(domain.discrete_actions, **params)

class SoftMax_mixture(Policy):
    def __init__(self, valuefns, policies, **argk):
        super(SoftMax_mixture, self).__init__()
        self.valuefns = valuefns
        self.policies = policies
        self.values = np.zeros(len(valuefns))
        self.temp = argk.get('temperature', 0.1)

    def __call__(self, state):
        self.values[:] = [vfn(state) for vfn in self.valuefns]
        self.values /= self.temp
        ev = np.exp(self.values)
        return self.policies[weighted_values(ev/np.sum(ev))](state)

# class linearSoftMax_mixture(Policy):
#     def __init__(self, valuefn, policies, **argk):
#         super(SoftMax_mixture, self).__init__()
#         self.value_fn = valuefn
#         self.policies = policies
#         self.temp = argk.get('temperature', 0.1)
#
#     def __call__(self, state):
#         if issubclass(state, np.uint):
#             values = np.sum(self.value_fn.theta[:,state], axis=1)
#         else:
#             values = self.value_fn.theta.dot(state)
#         values /= self.temp
#         ev = np.exp(values)
#         return self.policies[weighted_values(ev/np.sum(ev))](state)

class Max_mixture(Policy):
    def __init__(self, valuefns, policies, **argk):
        super(SoftMax_mixture, self).__init__()
        self.valuefns = valuefns
        self.policies = policies
        self.values = np.zeros(len(valuefns))

    def __call__(self, state):
        self.values[:] = [vfn(state) for vfn in self.valuefns]
        return self.policies[np.argmax(self.values)](state)

def weighted_values(probabilities, size=1):
    bins = np.cumsum(probabilities)
    return np.digitize(random_sample(size), bins)

# def policy_evaluation(rewards,
#                       gamma,
#                       policy,
#                       environment,
#                       method = 'LSTDlambda',
#                       **args):
#
#     if method == 'LSTDlambda':
#         return [LSTDlambda(policy,
#                            environment,
#                            gamma,
#                            r,
#                            **args) for r in rewards]


def evaluate_policy(domain, policy, iterations = 1):
    rew = 0.0
    for _ in xrange(iterations):
        r_t, s_t = domain.reset()
        rew += r_t
        while s_t is not None:
            r_t, s_t = domain.step(policy(s_t))
            rew += r_t
    return rew/iterations

def PGPE(x0,
         evaluator,
         policy_generator,
         sigma0 = 0.5,
         max_iterations = 1000,
         evaluation_iter = 1,
         final_eval_iter = 10,
         alpha_mu = 0.1,
         alpha_sigma = 0.1,
         alpha_baseline = 0.1):

    if not isinstance(alpha_mu, np.ndarray):
        alpha_mu = np.ones_like(x0) * alpha_mu

    if not isinstance(alpha_sigma, np.ndarray):
        alpha_sigma = np.ones_like(x0) * alpha_sigma

    if not isinstance(sigma0, np.ndarray):
        sigma = np.ones_like(x0) * sigma0
    else:
        sigma = sigma0

    x = x0

    baseline = evaluator(policy_generator(x0), final_eval_iter)

    max_val = float('-inf')
    rewards = [baseline]

    for i in xrange(max_iterations):
        perturb = np.random.normal(0, sigma)

        x_plus = x + perturb
        x_minus = x - perturb

        val_plus = evaluator(policy_generator(x_plus), evaluation_iter)
        val_minus = evaluator(policy_generator(x_minus), evaluation_iter)

        mu_rew = 0.5 * (val_plus + val_minus)
        d_rew = val_plus - val_minus
        max_val = np.max((val_plus, val_minus, max_val))

        baseline = baseline * (1.0 - alpha_baseline) + alpha_baseline * mu_rew

        if np.abs(max_val - mu_rew) > 1e-8:
            d_mu = 0.5 * perturb * d_rew / (max_val - mu_rew)
        else:
            d_mu = np.zeros_like(x)

        if np.abs(max_val - baseline) > 1e-8:
            d_sigma = ((mu_rew - baseline) * (perturb**2 - sigma**2) / sigma)\
                            / (max_val - baseline)
        else:
            d_sigma = np.zeros_like(x)

        x += alpha_mu * d_mu
        sigma += alpha_sigma * d_sigma
        sigma = np.maximum(sigma, 1e-2)

        rewards.append(evaluator(policy_generator(x), final_eval_iter))
    return x, rewards


