import numpy as np
from policy import weighted_values

class Agent(object):
    def __init__(self):
        pass

    def step(self, r, state):
        pass

    def reset(self):
        pass

    def proposeAction(self, state):
        pass


class POSarsa(Agent):
    def __init__(self, policy, valuefn, tracker, **argk):
        self.policy = policy
        self.valuefn = valuefn
        self.s_t = None
        self.a_t = None
        self.tracker = tracker

    def step(self, r, s_tp1):
        s_tp1 = self.tracker.update(s_tp1)
        if s_tp1 != None:
            a_tp1 = self.policy(s_tp1)
        else:
            a_tp1 = None
        self.valuefn.update(self.s_t, self.a_t, r, s_tp1, a_tp1)

        self.s_t = s_tp1
        self.a_t = a_tp1

        return a_tp1

    def reset(self):
        self.s_t = None
        self.a_t = None
        self.tracker.reset()

    def getActor(self):
        # actor generated will change its behaviour if the agent's value fn is
        # changed
        return POActor(self.tracker.copy(), self.policy)

class POActor(object):
    def __init__(self, tracker, policy):
        self.tracker = tracker
        self.policy = policy

    def reset(self):
        self.tracker.reset()

    def proposeAction(self, state):
        s = self.tracker.update(state)
        return self.policy(s)



class Sarsa(Agent):
    def __init__(self, policy, valuefn, **argk):
        self.policy = policy
        self.valuefn = valuefn
        self.s_t = None
        self.a_t = None

    def step(self, r, s_tp1):
        if s_tp1 != None:
            a_tp1 = self.policy(s_tp1)
        else:
            a_tp1 = None
        self.valuefn.update(self.s_t, self.a_t, r, s_tp1, a_tp1)

        self.s_t = s_tp1
        self.a_t = a_tp1

        return a_tp1

    def reset(self):
        self.s_t = None
        self.a_t = None

    def proposeAction(self, state):
        return self.policy(state)

class TabularActionSarsa(Agent):
    def __init__(self, actions, policy, valuefn, **argk):
        self.policy = policy
        self.valuefn = valuefn
        self.s_t = None
        self.a_t = None
        self.actions = actions

    def step(self, r, s_tp1):
        if s_tp1 != None:
            a_tp1 = self.policy(s_tp1)
        else:
            a_tp1 = None
        self.valuefn.update(self.s_t, self.a_t, r, s_tp1, a_tp1)

        self.s_t = s_tp1
        self.a_t = a_tp1

        return self.actions[a_tp1] if a_tp1 != None else None

    def reset(self):
        self.s_t = None
        self.a_t = None

    def proposeAction(self, state):
        return self.actions[self.policy(state)]

class LinearTabularPolicySarsa(Agent):
    def __init__(self, actions, mix_policy, policies, valuefn, **argk):
        self.policy = mix_policy
        self.policies = policies
        self.valuefn = valuefn
        self.s_t = None
        self.rho = None
        self.actions = actions

    def step(self, r, s_tp1):
        if s_tp1 != None:
            p_pi = self.policy.getprob(s_tp1)
            pi_tp1 = weighted_values(p_pi)
            p_a = np.vstack([p.getprob(s_tp1) for p in self.policies])
            a_tp1 = weighted_values(p_a[pi_tp1,:])[0]
            rho_tp1 = p_a[:,a_tp1]/ p_a[:,a_tp1].dot(p_pi)
        else:
            a_tp1 = None
            rho_tp1 = None
            a_tp1 = None
        self.valuefn.update(self.s_t, r, s_tp1, self.rho)

        self.s_t = s_tp1
        self.rho = rho_tp1

        return self.actions[a_tp1] if a_tp1 != None else None

    def reset(self):
        self.s_t = None

    def proposeAction(self, state):
        return self.actions[self.policy(state)]


class Sarsa_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        valuefn = params.get('valuefn')
        policy = params.get('policy')
        return Sarsa(**params)