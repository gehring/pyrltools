import numpy as np
from itertools import product, izip, repeat, imap, chain

class trial_monitor(object):
    def __init__(self, **params):
        self.score = []

    def start(self):
        self.score = []

    def update(self, r, *args, **karg):
        self.score.append(r)

    def getscore(self):
        return self.score

class trial_monitor_factory(object):
    def __init__(self, **param):
        self.param = param
    def __call__(self, **args):
        params = dict(self.param)
        params.update(args)
        return trial_monitor(**params)

class evaluation_monitor(object):
    def __init__(self, num_eval, evaluator, **param):
        self.score = []
        self.num_eval = num_eval
        self.evaluator = evaluator

    def start(self):
        self.score = []

    def update(self, r, s_t, domain, agent,  *args, **karg):
        self.score.append(np.mean(
                [self.evaluator(domain, agent) for i in xrange(self.num_eval)]))

    def getscore(self):
        return self.score

class evaluation_monitor_factory(object):
    def __init__(self, **param):
        self.param = param
    def __call__(self, **args):
        params = dict(self.param)
        params.update(args)
        return evaluation_monitor(**params)

class interval_monitor(object):
    def __init__(self, eval_interval, monitor, **param):
        self.eval_interval = eval_interval
        self.monitor = monitor
        self.count = 0
    def start(self):
        self.count = 0
        self.monitor.start()

    def update(self,  *args, **karg):
        if self.count % self.eval_interval ==  0:
            self.monitor.update(*args, **karg)

    def getscore(self):
        return self.score

class interval_monitor_factory(object):
    def __init__(self, factory, **param):
        self.factory = factory
        self.param = param
    def __call__(self, **args):
        params = dict(self.param)
        params.update(args)
        return interval_monitor(monitor = self.factory(**params), **params)

class bundled_monitor(object):
    def __init__(self, monitors, **param):
        self.monitors = monitors

    def start(self):
        for m in self.monitors:
            m.start()

    def update(self,  *args, **kargs):
        for m in self.monitors:
            m.update( *args, **kargs)

    def getscore(self):
        return [m.getscore() for m in self.monitors]

class bundled_monitor_factory(object):
    def __init__(self, factories, **param):
        self.factories = factories
        self.param = param
    def __call__(self, **args):
        params = dict(self.param)
        params.update(args)
        return bundled_monitor(monitors = [f(**params) for f in self.factories])

def evaluate_trial(domain, agent):
    domain_copy = domain.copy()
    r, s_t = domain_copy.reset()
    cum_rew = r
    while s_t != None:
        r, s_t = domain_copy.step(agent.proposeAction(s_t))
        cum_rew += r

    return cum_rew

def evaluate_1000_steps(domain, agent):
    domain_copy = domain.copy()
    r, s_t = domain_copy.reset()
    cum_rew = r
    for i in xrange(1000):
        if s_t == None:
            agent.reset()
            r, s_t = domain_copy.reset()
        else:
            r, s_t = domain_copy.step(agent.proposeAction(s_t))
        cum_rew += r

    return cum_rew

def train_steps_agent(domain, agent, monitor, num_train_steps, num_eval, eval_interval, **args):
    r, s_t = domain.reset()
    agent.reset()
    monitor.start()
    for i in xrange(num_train_steps):
        monitor.update(r, s_t, domain, agent)
        if s_t == None:
            agent.step(r, s_t)
            agent.reset()
            r, s_t = domain.reset()
        else:
            r, s_t = domain.step(agent.step(r, s_t))
    monitor.update(r, s_t, domain, agent)
    return monitor.getscore()

def train_trials_agent(domain, agent, monitor, num_train_steps, num_eval, eval_interval, **args):
    for i in xrange(num_train_steps):
        r, s_t = domain.reset()
        agent.reset()
        monitor.start()
        cum_r = r
        while s_t != None:
            r, s_t = domain.step(agent.step(r, s_t))
            cum_r += r
        agent.step(r,s_t)
        monitor.update(cum_r, s_t, domain, agent)
    return monitor.getscore()

def train_agent(**args):
    domain_factory = args.get('domain_factory')
    projector_factory = args.get('projector_factory')
    policy_factory = args.get('policy_factory')
    value_fn_factory = args.get('valuefn_factory')
    monitor_factory = args.get('monitor_factory')
    agent_factory = args.get('agent_factory')
    agent_trainer = args.get('trainer', train_steps_agent)
    key_parameter = args.get('key_parameters')

    domain = domain_factory(**args)
    projector = projector_factory(domain = domain, **args)
    param = dict(args)
    param['layers'] = [projector.size] + args.get('internal_layers', [40]) + [1]
    valuefn = value_fn_factory(projector = projector, **param)
    policy = policy_factory(domain = domain, valuefn = valuefn, **args)
    agent = agent_factory(policy = policy, valuefn = valuefn, **args)

    arguments = dict(args)
    arguments['domain'] = domain
    arguments['agent'] = agent
    arguments['monitor'] = monitor_factory(**args)
    return tuple([ args[k] for k in key_parameter]), agent_trainer(**arguments)

def getAllRuns(product_param):
    # returns the product of every parameter as dict then generates every independent run
    num_runs = product_param['num_runs'][0]
    return chain(*(imap(dict,product(*(izip(repeat(k),v) for k,v in product_param.iteritems()))) for i in xrange(num_runs)))

