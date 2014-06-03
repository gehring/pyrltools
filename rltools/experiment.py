import numpy as np
from itertools import product, izip, repeat, imap, chain

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

def evaluateAgent(domain, agent, evaluator, num_trials):
    return [ evaluator(domain, agent) for i in range(num_trials)]

def train_steps_agent(domain, agent, evaluator, num_train_steps, num_eval, eval_interval, **args):
    score= []
    r, s_t = domain.reset()
    agent.reset()
    for i in xrange(num_train_steps):
        if i % eval_interval == 0:
            score.append( np.mean(evaluateAgent(domain, agent, evaluator, num_eval)))

        if s_t == None:
            agent.step(r, s_t)
            agent.reset()
            r, s_t = domain.reset()
        else:
            r, s_t = domain.step(agent.step(r, s_t))
    return score

def train_trials_agent(domain, agent, evaluator, num_train_steps, num_eval, eval_interval, **args):
    score= []
    for i in xrange(num_train_steps):
        if i % eval_interval == 0:
                score.append( np.mean(evaluateAgent(domain, agent, evaluator, num_eval)))
        r, s_t = domain.reset()
        agent.reset()
        while s_t != None:
            r, s_t = domain.step(agent.step(r, s_t))
        agent.step(r,s_t)

    return score

def train_score_per_trial_agent(domain, agent, num_train_steps, **args):
    score= np.empty(num_train_steps, dtype = np.double)
    for i in xrange(num_train_steps):
        r, s_t = domain.reset()
        agent.reset()
        score[i] = r
        while s_t != None:
            r, s_t = domain.step(agent.step(r, s_t))
            score[i] += r
        agent.step(r,s_t)

    return score

def train_agent(**args):
    domain_factory = args.get('domain_factory')
    projector_factory = args.get('projector_factory')
    policy_factory = args.get('policy_factory')
    value_fn_factory = args.get('valuefn_factory')
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
    return tuple([ args[k] for k in key_parameter]), agent_trainer(**arguments)

def getAllRuns(product_param):
    # returns the product of every parameter as dict then generates every independent run
    num_runs = product_param['num_runs'][0]
    return chain(*(imap(dict,product(*(izip(repeat(k),v) for k,v in product_param.iteritems()))) for i in xrange(num_runs)))

