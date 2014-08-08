import numpy as np

class Policy(object):
    def __init__(self):
        pass

    def __call__(self, state):
        pass

class Egreedy(Policy):
    def __init__(self, actions, valuefn, **argk):
        super(Egreedy, self).__init__()
        self.actions = actions
        self.value_fn = valuefn
        self.epsilon = argk.get('epsilon', 0.1)

    def __call__(self, state):
        values = np.array([self.value_fn(state, act) for act in self.actions])
        if np.all(np.isnan(values)):
            values[:] = 1.0/len(self.actions)
        else:
            m = np.nanargmax(values)
            values[:] = self.epsilon/len(self.actions)
            values[m] += 1-self.epsilon
        a = np.random.choice(range(len(self.actions)), p=values)
        return self.actions[a]

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

    def __call__(self, state):
        values = np.array([self.value_fn(state, act) for act in self.actions])
        if np.any(np.isnan(values)):
            values[:] = 1.0/len(self.actions)
        else:
            values /= self.temp
            values = np.exp(values)
            values /= np.sum(values)
        a = np.random.choice(range(len(self.actions)), p=values)
        return self.actions[a]

class SoftMax_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        domain = params.get('domain')
        return SoftMax(domain.discrete_actions, **params)

# class GradientDescentPolicy(Policy):
#     def __init__(self, actions, value_fn, **argk):
#         super(Egreedy, self).__init__()
#         self.actions = actions.copy()
#         self.value_fn = value_fn
#
#         self.epsilon = argk.get('epsilon', 0.1)