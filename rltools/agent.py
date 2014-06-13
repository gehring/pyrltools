class Agent(object):
    def __init__(self):
        pass

    def step(self, r, state):
        pass

    def reset(self):
        pass

    def proposeAction(self, state):
        pass


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


class Sarsa_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        valuefn = params.get('valuefn')
        policy = params.get('policy')
        return Sarsa(**params)