import numpy as np

from rltools.valuefn import solveKBRL
from scipy.constants.constants import psi

class KBRLRRT(object):
    def __init__(self, env, heuristic, sampler, bias, psi):
        # The environment, it is assumed that, given a scoring function and a
        # set of samples, it can produce the best next point.
        self.env = env
        
        # This is the uninformed heuristic. It works and is used like any other
        # so any intuition about which to use should be unchanged.
        # The optimal heuristic is the optimal cost-to-go.
        self.heuristic = heuristic
        
        # Sampler is a callable which generates random samples in the free
        # space.
        self.sampler = sampler
        
        # This value serves to bias the approx. cost-to-go to use the heuristic
        # when similarities between observed samples are low.
        # The larger this is, the more aggressive the bias is.
        self.bias = bias
        
        # psi is the similarity function comparing two states
        # it is assumed that if only ones of its arguments is an array
        # it will return a vector of similarities.
        # If both are arrays, it will return all pairs distances with the 
        # first argument varying with the rows.
        self.psi = psi
    
    def plan(self, 
             start,
             goal, 
             goal_test, 
             max_iterations, 
             screenshot_rate=1,
             save_screenshot=False):
        env = self.env
        heuristic = self.heuristic
        sampler = self.sampler
        
        # sample new point to boot strap the process
        point = sampler()
        h_hat = lambda x: heuristic(x, point)
#         h_hat = lambda x: heuristic(x, goal)
        origin, cost, next_point = env.get_best([start], h_hat)
        
        # initialize the sample set
        samples = (np.array([start], dtype='float32'),
                   np.array([cost], dtype='float32'),
                   np.array([next_point], dtype='float32'))
        
        # initialize the parent pointer tree structure
        parents = {tuple(next_point):tuple(start)}
        
        # all nodes
        nodes = [start, next_point]
        
        # set of nodes with children
        has_child = set()
        has_child.add(tuple(start))
        count = 0
        failed = False
        screenshots = []
        
        while not goal_test(next_point):
            # sample random destination point
            point = sampler()

            

            # if asked, take a screenshot of the state
            if save_screenshot and (count % screenshot_rate) == 0:
                vpc = self.solve_values_plus_cost(samples, goal)
                h_hat = approx_cost_to_go(goal, 
                                     self.psi, 
                                     vpc.copy(), 
                                     self.bias, 
                                     heuristic, 
                                     [samples[i].copy() for i in range(3)])   
                screenshots.append( (parents.copy(), h_hat))


            # approximate cost-to-go heuristic
            vpc = self.solve_values_plus_cost(samples, point)
            # update heuristic
            h_hat = approx_cost_to_go(point, 
                                     self.psi, 
                                     vpc, 
                                     self.bias, 
                                     heuristic, 
                                     samples)   
            # approximate cost-to-go heuristic
#                 vpc = self.solve_values_plus_cost(samples, goal)
#                 # update heuristic
#                 h_hat = approx_cost_to_go(goal, 
#                                          self.psi, 
#                                          vpc, 
#                                          self.bias, 
#                                          heuristic, 
#                                          samples) 
            
            
            # find the best expansion
            origin, cost, next_point = env.get_best(nodes, h_hat)
            
            
            # if origin is not in the samples, add it
            if tuple(origin) not in has_child:
#                 samples = ( np.vstack((samples[0], [origin])),
#                             np.hstack((samples[1], [cost])),
#                             np.vstack((samples[2], [next_point])))
                has_child.add(tuple(origin))
                
            samples = ( np.vstack((samples[0], [origin])),
                            np.hstack((samples[1], [cost])),
                            np.vstack((samples[2], [next_point])))
#             has_child.add(tuple(origin))
            
            # add new point to the parent pointer tree
            parents[tuple(next_point)] = tuple(origin)
            nodes.append(next_point)
            count += 1
            print count
            
            
            if count >= max_iterations:
                failed = True
                break
        
        # extract the path from the parent pointer tree
        if not failed:
            path = self.generate_path(parents, next_point)
            path.append(goal)
        else:
            path = None
            
        
        # process the last screenshot
        vpc = self.solve_values_plus_cost(samples, goal)
        h_hat = approx_cost_to_go(goal, 
                                     self.psi, 
                                     vpc.copy(), 
                                     self.bias, 
                                     heuristic, 
                                     [samples[i].copy() for i in range(3)])    
        screenshots.append((parents, h_hat))
        
        
        return path, {'start':tuple(start),
                      'goal': tuple(goal),
                      'path':path,
                      'environment': env,
                      'screenshots': screenshots}
    
    def generate_path(self, parents, point):
        path = [point]
        
        # go up the parent pointer tree to the root
        while tuple(point) in parents:
            point = parents[tuple(point)]
            path.append(point)
        path.reverse()
        return path
        
        
    def solve_values_plus_cost(self, samples, goal):
        # find how close each sample is to the goal
        atgoal = self.psi(samples[2], goal)
        
        # compute the similarity matrix 
        K = self.psi(samples[2], samples[0])
        
        # compute mass with bias
        if K.ndim <1:
            K = K[np.newaxis, np.newaxis]
        mass = np.sum(K, axis=1)
#         print samples
#         print mass
#         print K
#         print atgoal
        mass += atgoal
        mass += self.bias
        
        # compute heuristic bias
        epsilon = self.bias/mass
        eta = self.heuristic(samples[2], goal)
        
        vpc = solveKBRL(K/mass[:,None], samples[1], epsilon*eta) + samples[1]
        if vpc.ndim < 2:
            vpc = vpc.reshape((-1,1))
        return vpc
    
def compute_h_hat(x, goal, psi, vpc, bias, heuristic, samples):
    atgoal = psi(x, goal)
    
    k = psi(x, samples[0])
    
    
#     print 'k', k.shape
#         print 'atgoal', atgoal.shape
#         
    if k.ndim>1:
        atgoal = atgoal[:,None]
    
    mass = np.sum(k) + atgoal + bias
    k /= mass
    
    epsilon = bias/mass
    eta = heuristic(x, goal)
    
    if k.ndim < 2:
        k = k.reshape((-1,1))
    
#         print 'other'    
#         print epsilon.shape
#         print eta.shape
#         print mass.shape
#         print vpc.shape
#         
#         
#         print k.dot(vpc).squeeze()
#         print eta*epsilon.squeeze()
    return k.dot(vpc).squeeze() + eta*epsilon.squeeze()
        
class approx_cost_to_go(object):
    def __init__(self, goal, psi, vpc, bias, heuristic, samples):
        self.goal = goal
        self.psi = psi
        self.vpc = vpc
        self.bias = bias
        self.heuristic = heuristic
        self.samples = samples
    
    def __call__(self, x):
        return compute_h_hat(x, 
                             self.goal,
                             self.psi,
                             self.vpc,
                             self.bias,
                             self.heuristic,
                             self.samples)

class RBF_Kernel(object):
    def __init__(self, width):
        self.w = np.sqrt(width)[None,:,None]
        
    def __call__(self, x, y):
        w = self.w
        
        if x.ndim == 1:
            x = x[None, :]
        if y.ndim == 1:
            y = y[None, :]
        
        d = -(((x[:,:, None]-y.T[None, :, :])/w)**2).sum(axis=1)
        return np.exp(d).squeeze()
        

