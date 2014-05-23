from MountainCar import MountainCar
from agent import Sarsa
from policy import Egreedy
from valuefn import  NeuroSFTD
from representation import FlatStateAction, Normalizer
import numpy as np

def evaluate_trial(domain, agent):
    domain_copy = domain.copy()
    r, s_t = domain_copy.reset()
    cum_rew = r
    while s_t != None:
        r, s_t = domain_copy.step(agent.proposeAction(s_t))
        cum_rew += r

    return cum_rew

def evaluateAgent(domain, agent, num_trials):
    return [ evaluate_trial(domain, agent) for i in range(num_trials)]

def train_agent(domain, agent, num_steps, num_eval, eval_interval):
    score= []
    r, s_t = domain.reset()
    for i in xrange(num_steps):
        if i % eval_interval == 0:
            score.append( np.mean(evaluateAgent(domain, agent, num_eval)))

        if s_t == None:
            agent.reset()
            r, s_t = domain.reset()
        else:
            r, s_t = domain.step(agent.step(r, s_t))
    return score


def getRuns(**args):
    alpha = args.get('alpha')
    eta = args.get('eta')
    epsilon = args.get('epsilon')
    gamma = args.get('gamma')
    mommentum = args.get('mommentum')
    num_runs = args.get('num_runs', 10)

    param = {'alpha':alpha,
             'eta':eta,
             'epsilon':epsilon,
             'gamma':gamma,
             'mommentum':mommentum}

    for i in range(num_runs):
        domain = MountainCar(random_start=True, max_episode = 1000)
        projector = Normalizer(FlatStateAction(2,1),
                               domain.state_range,
                               domain.action_range)
        param['layers'] = [projector.size] + args.get('layers', [40]) + [1]
        valuefn = NeuroSFTD(projector, **param)
        policy = Egreedy( domain.discrete_actions, valuefn)
        agent = Sarsa(policy, valuefn)
        yield domain, agent

def get_score_list(**args):
    num_train_steps = args.get('num_train_steps')
    num_eval_trial = args.get('num_eval_trial')
    eval_interval = args.get('eval_interval')
    return map(lambda (d,a): train_agent(d, a, num_train_steps,
                                                    num_eval_trial,
                                                    eval_interval),
                            getRuns(**args))


if __name__ == '__main__':

    param = {   'alpha' : 0.09 ,
                'gamma' : 0.9,
                'eta' : 0.4,
                'epsilon' : 0.05,
                'num_runs' : 1,
                'layers': [400]}
    for domain, agent in getRuns(**param):
        r, s_t = domain.reset()
        last = 0
        for i in xrange(100000):
            if s_t == None:
                agent.reset()
                r, s_t = domain.reset()
                print i - last
                last = i
            else:
                r, s_t = domain.step(agent.step(r, s_t))

