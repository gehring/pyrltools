from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.npplanning import sample_gaussian
from rltools.planner import SingleEmbeddedAgent

#from matplotlib import collections  as mc

from itertools import izip

#import matplotlib.pyplot as plt
import numpy as np

def run_exp(i):
    print i 
    from rltools.MountainCar import MountainCar, PumpingPolicy
    from rltools.npplanning import sample_gaussian
    from rltools.planner import SingleEmbeddedAgent
    
    
    from itertools import izip
    
    import numpy as np
    
    np.random.seed(i*123123)
    
                
    def plot_samples(S, Sp):
        lines = [ [x0, x1] for x0,x1 in zip(S, Sp) if x1 is not None]
        
        sample_color = [(0,0,1.0,0.8)]*len(lines)
        
    #     print traj, len(traj)
        lc = mc.LineCollection(lines, color = sample_color, linewidths=2)
        plt.gca().add_collection(lc)
    
    def get_next_state(domain, state, action):
        domain.reset()
        domain.state = state.copy()
        return domain.step(action)[1]
        
    def display_arrows(states, next_states):
        ax = plt.gca()
        for s_t, s_tp1 in izip(states, next_states):
            if s_tp1 is not None:
                ax.arrow(s_t[0], s_t[1], s_tp1[0] -s_t[0], s_tp1[1]- s_t[1], head_length = 0.02, fc = 'k', ec='k', width=0.0001)
    
    def fourier_features(X, w):
        if X.ndim == 1:
            X = X.reshape((1,-1))
        features = np.hstack((X, np.zeros((X.shape[0], 1)))).dot(w)        
        features = np.hstack((np.sin(features), np.cos(features), np.ones((features.shape[0],1))))
        return features.squeeze()/ np.sqrt(w.shape[1])
    
    def run_episode(domain, agent, max_length = 1000):
        x_tp1 = domain.reset()
        x_t, a_t, r_t = None, None, None
    
        count = 0
        traj = []
        agent.reset()
        while x_tp1 is not None and count < max_length:
            count += 1
            agent.step(x_t, a_t, r_t, x_tp1)
            a_t = agent.get_action()
            x_t = x_tp1
            r_t, x_tp1 = domain.step(a_t)
            traj.append((x_t, a_t, r_t, x_tp1))
        agent.step(x_t, a_t, r_t, x_tp1)
        return traj
    
    def generate_data(domain, policy, num_traj, average = False):
        samples = []
        actions = domain.discrete_actions
        sampled_traj = []
        for i in xrange(num_traj-1):
            traj = []
            s_t = domain.reset()
            while s_t is not None:
                if np.random.rand(1) < 0.05:
                    a = 1#np.random.randint(3)
                else:
                    a = 2 if s_t[1] > 0 else 0
                r_t, s_tp1 = domain.step(actions[a])
                traj.append((s_t, actions[a], r_t, s_tp1))
                samples.append((s_t, actions[a], r_t, s_tp1))
                s_t = s_tp1
            sampled_traj.append(traj)
            
        domain.random_start= False        
        s_t = domain.reset()
        traj = []
        while s_t is not None:
            if np.random.rand(1) < 0.1:
                a = 1#np.random.randint(3)
            else:
                a = 2 if s_t[1] > 0 else 0
            r_t, s_tp1 = domain.step(actions[a])
            traj.append((s_t, actions[a], r_t, s_tp1))
            samples.append((s_t, actions[a], r_t, s_tp1))
            s_t = s_tp1
        sampled_traj.append(traj)
        
        X_t = []
        A_t = []
        X_tp1 = []
        X_term = []
        A_term = []
        R_t = []
        R_term = []
        for traj in sampled_traj:
            for s_t, a, r_t, s_tp1 in traj:
                if s_tp1 is None:
                    X_term.append(s_t)
                    R_term.append(r_t)
    #                R_term.append(1)
                    A_term.append(a)
                else:
                    X_t.append(s_t)
                    R_t.append(r_t)
    #                R_t.append(0)
                    A_t.append(a)
                    X_tp1.append(s_tp1)
        return (np.array(X_t), np.array(A_t), np.array(R_t), np.array(X_tp1),
                np.array(X_term), np.array(A_term), np.array(R_term), samples)
    
    #################################################
    
    # initialize the domain
    domain = MountainCar(random_start = True,
    					max_episode = np.Infinity)
    
    state_range = domain.state_range
    actions = domain.discrete_actions
    
    arrow_grid = 10
    x = np.linspace(domain.state_range[0][0], domain.state_range[1][0], arrow_grid)
    y = np.linspace(domain.state_range[0][1], domain.state_range[1][1], arrow_grid)
    X, Y = np.meshgrid(x, y)
    states = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
    next_state=[]
    for a in xrange(3):
        next_state.append([])
        for s in states:
            next_state[a].append(get_next_state(domain, s, domain.discrete_actions[a]))
    
    # define kernel over actions
    def single_action_kernel(i):
        #return lambda b: np.exp(-np.sum(((actions[i][None,:]-b)/6.0)**2, axis=1)/(0.5**2))
        return lambda b: (actions[i][None,:] == b).astype('float').squeeze()
    
    action_kernels = [ single_action_kernel(i) for i in xrange(len(actions))]
    
    # initialize the representation
    width = np.array([0.2, 0.2])
    scale = ((state_range[1] - state_range[0]) * width)
    
    num_gauss = 2000
    w = sample_gaussian(state_range[0].shape[0], num_gauss, scale)   
    phi = lambda X: fourier_features(X, w)
    
    #X_t, A_t, R_t, X_tp1, X_term, A_term, R_term, samples = generate_data(domain, PumpingPolicy(), 1, False)
    X_t, A_t, R_t, X_tp1, X_term, A_term, R_term, samples = [None] * 8
    
    agent = SingleEmbeddedAgent(plan_horizon = 150,
                     dim = num_gauss*2+1, 
                     X_t = X_t,
                     A_t = A_t, 
                     X_tp1 = X_tp1, 
                     R_t = R_t,
                     X_term = X_term,
                     A_term = A_term,
                     R_term = R_term,
                     max_rank = 50,
                     blend_coeff = 0.1,
                     phi = phi,
                     action_kernels = action_kernels,
                     actions = actions,
                     learning_rate = 0.005,
                     discount = 0.99,
                     lamb = 0.2,
                     update_models = True,
                     use_valuefn = True,
                     use_diff_model = False)
    
    traj = []
    num_episodes = 25
#    f = plt.figure()
    
    num_points = 100
    x = np.linspace(domain.state_range[0][0], domain.state_range[1][0], num_points)
    y = np.linspace(domain.state_range[0][1], domain.state_range[1][1], num_points)
    X, Y = np.meshgrid(x, y)
    
    
    
    domain.random_start = False
    for i in xrange(num_episodes):
        traj.append(run_episode(domain, agent))
        print np.sum(t[2] for t in traj[-1])
#        
#        
#        vals = phi(np.hstack((X.reshape((-1,1)),Y.reshape((-1,1))))).dot(agent.theta)
#        plt.subplot(num_episodes/5, 5, i+1)
#        plt.gca().set_xlim([state_range[0][0], state_range[1][0]])
#        plt.gca().set_ylim([state_range[0][1], state_range[1][1]])
#        c = plt.pcolormesh(X, Y, vals.reshape((num_points, -1)), cmap='Oranges')
#        Xt, A, R, Xtp1 = zip(*traj[-1])
#        display_arrows(states, next_state[a])
#        plot_samples(Xt, Xtp1)
#        f.colorbar(c)
#        plt.show()
        
    return traj

run_exp(0)
##################################################
#from multiprocessing import Pool
#import pickle
#
#results = []
#filename = 'test-horizon-50-r-50-lamb-02-alp-0005-blend-01-disc-099-decay-02.data'
#def results_callback(res):
#    results.append(res)
#    with open(filename, 'wb') as f:
#        pickle.dump(results, f)
#        f.close()
#
#
#pool = Pool(12)
#print 'starting'
#
#for i in xrange(120):
#    pool.apply_async(run_exp, (i,), callback = results_callback)
#pool.close()
#pool.join()
#
#print 'done'
###############################################
