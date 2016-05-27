# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 12:31:35 2016

@author: cgehri
"""
import numpy as np
from rltools.model import CompressedModel, SingleCompressModel
from rltools.npplanning import create_old_embed, find_stoc_plan


class PlanningAgent:
    def __init__(self):
        raise NotImplementedError('Should be implemented by subclass')
    
    def get_next_action(self):
        raise NotImplementedError('Should be implemented by subclass')
    
    def step(self, x_t, a_t, r_t, x_tp1):
        raise NotImplementedError('Should be implemented by subclass')
        
class LEMAgent(PlanningAgent):
    def __init__(self,
                 plan_horizon,
                 dim,
                 num_actions,
                 Xa_t,
                 Xa_tp1,
                 Ra,
                 Xa_term,
                 Ra_term,
                 blend_coeff,
                 phi,
                 use_average_reward = False,
                 lamb = 0.2,
                 discount = 0.99,
                 learning_rate = 0.01):

        self.use_average_reward = use_average_reward
        self.H = plan_horizon
        self.learning_rate = learning_rate
        self.discount = discount

        self.plan = np.ones((self.H, num_actions+1))/(num_actions+1)
        self.plan[0,:] = 1.0/num_actions
        self.plan[0,-1] = 0

        self.phi = phi
        self.blend_coeff = blend_coeff
        self.num_actions = num_actions + 1
        self.lamb = lamb
        phia_t = []
        phia_tp1 = []
        all_Ra = []

        self.alphas = np.zeros((self.H+1, dim))
        self.betas = np.zeros((self.H, dim))

        for a in xrange(num_actions):
            phi_t = phi(np.vstack((Xa_t[a], Xa_term[a])))
            phi_tp1 = np.vstack((phi(Xa_tp1[a]), np.zeros((Xa_term[a].shape[0], phi_t.shape[1]))))
            phia_t.append(phi_t)
            phia_tp1.append(phi_tp1)
            all_Ra.append(np.hstack((Ra[a], Ra_term[a])))

        if use_average_reward:
            avgrew = np.mean(np.hstack(all_Ra))

            for a in xrange(num_actions):
                all_Ra[a] -= avgrew

        models = [ self.solve_model(phi_t, phi_tp1, r) for phi_t, phi_tp1, r in zip(phia_t, phia_tp1, all_Ra)]
        models.append((np.zeros((dim,dim)), np.zeros(dim)))
        Fa, wa = zip(*models)
        self.models = list(Fa), list(wa)

    def solve_model(self, phi_t, phi_tp1, r):
        print phi_t.shape, phi_tp1.shape
        U,S,Vt = np.linalg.svd(phi_t, full_matrices=False)
        S = S/(S**2 + self.lamb)
        print U.shape, S.shape, Vt.shape

        F = phi_tp1.T.dot(U.dot(np.diag(S).dot(Vt)))
        w = r.dot(U).dot(np.diag(S).dot(Vt))
        return F, w


    def step(self, x_t, a_t, r_t, x_tp1):
            self.x_t = x_tp1
            self.plan[:-1,:] = self.plan[1:,:]
            self.plan[-1,:] = 1.0/self.plan.shape[1]
            self.plan = self.plan*0.5 + 0.5/self.plan.shape[1]
            self.plan[0,-1] = 0.0
            self.plan[0,:] /= np.sum(self.plan[0,:])
            self.improve_plan()

            if x_t is not None:
                phi_t = self.phi(x_t)
                delta =  self.plan_val - self.models[1][-1].dot(phi_t)
                self.models[1][-1] += self.learning_rate * delta * phi_t
            
        
    def get_action(self, t = 0):
        return np.argmax(self.plan[t])

    def improve_plan(self):
            alphas = self.alphas
            betas = self.betas
            Fa, wa = self.models
            H = self.H
            k = self.num_actions
            alphas[H,:] =  wa[-1] # assuming terminal reward is zero if not using value function
            betas[0,:] = self.phi(self.x_t)
                
                
            self.plan, value = self.backward_step(alphas, betas, self.plan, improve = False)
            diff = np.Infinity
            
            while np.abs(diff) > np.abs(0.001*value):
                old_plan = self.plan.copy()
                self.plan, old_val = self.forward_step(alphas, betas, self.plan, improve = True)
                self.plan, new_val = self.backward_step(alphas, betas, self.plan, improve = True)
                diff = new_val - old_val
                print new_val
                if np.allclose(old_plan, self.plan, atol = 0.01):
                    break

            self.plan_val = new_val

    def backward_step(self, alphas, betas, plan, improve):
            k = self.num_actions
            H = self.H
            Fa, wa = self.models
            
            if improve:
                new_plan = plan.copy()
            else:
                new_plan = plan
                
            gamma = self.blend_coeff
            discount = self.discount

            vals = np.zeros(k)
            if improve:
                for a in xrange(k):
                    vals[a] = (discount*alphas[H,:].dot(Fa[a]) + wa[a]).dot(betas[H-1,:])
                a_best = np.argmax(vals)
                new_plan[H-1,:] *= gamma
                new_plan[H-1,a_best] += (1-gamma)
                
            for t in xrange(H-1, 0, -1):
                alphas[t,:] = sum( [ new_plan[t, b] * (discount*alphas[t+1,:].dot(Fa[b]) + wa[b]) for b in xrange(k)])
                for a in xrange(k):
                    vals[a] = (discount* alphas[t,:].dot(Fa[a]) + wa[a]).dot(betas[t-1,:])
                
                if improve:
                    if t == 1:
                        a_best = np.argmax(vals[:-1]) 
                    else:
                        a_best = np.argmax(vals) 
                    new_plan[t-1,:] *= gamma
                    new_plan[t-1,a_best] += (1-gamma)
                    

            new_val = vals.dot(new_plan[0])
            
            return new_plan, new_val

    def forward_step(self, alphas, betas, plan, improve):
            k = self.num_actions
            H = self.H
            Fa, wa = self.models
            
            if improve:
                new_plan = plan.copy()
            else:
                new_plan = plan
                
            gamma = self.blend_coeff
            discount = self.discount
            
            vals = np.zeros(k)
            for a in xrange(k):
                    vals[a] =(discount*alphas[1,:].dot(Fa[a]) + wa[a]).dot(betas[0,:])
            if improve:
                a_best = np.argmax(vals[:-1])
                new_plan[0,:] *= gamma
                new_plan[0,a_best] += (1-gamma)
                
            old_val = plan[0].dot(vals)
                
            for t in xrange(1, H):
                betas[t,:] = sum( [discount*new_plan[t-1, b] * Fa[b].dot(betas[t-1,:]) for b in xrange(k)])
                for a in xrange(k):
                    vals[a] =(discount*alphas[t+1,:].dot(Fa[a]) + wa[a]).dot(betas[t,:])
                    
                if improve:
                    a_best = np.argmax(vals)  
                    new_plan[t,:] *= gamma
                    new_plan[t,a_best] += (1-gamma)
                    
            return new_plan, old_val



class EmbeddedAgent(PlanningAgent):
    def __init__(self,
                 plan_horizon,
                 dim, 
                 num_actions, 
                 Xa_t, 
                 Xa_tp1, 
                 Ra,
                 Xa_term,
                 Ra_term,
                 max_rank,
                 blend_coeff,
                 phi,
                 update_models = False,
                 use_valuefn = False):
        self.phi = phi
        self.update_models = update_models
        self.blend_coeff = blend_coeff
        self.H = plan_horizon

        self.use_valuefn = use_valuefn
        if use_valuefn:
            self.theta = np.zeros(dim)
        
        phia_t = []
        phia_tp1 = []
        all_Ra = []
        for a in xrange(num_actions):
            phi_t = phi(np.vstack((Xa_t[a], Xa_term[a])))
            phi_tp1 = np.vstack((phi(Xa_tp1[a]), np.zeros((Xa_term[a].shape[0], phi_t.shape[1]))))
            phia_t.append(phi_t)
            phia_tp1.append(phi_tp1)
            all_Ra.append(np.hstack((Ra[a], Ra_term[a])))

        self.model = CompressedModel(dim = dim,
                                num_actions = num_actions,
                                Xa_t = phia_t,
                                Xa_tp1 = phia_tp1,
                                Ra = all_Ra,
                                max_rank = max_rank)
        self.plan = np.ones((self.H, num_actions))/num_actions
                    
        k = num_actions
        self.alphas = np.empty(k, dtype='O')
        self.betas = np.empty(k, dtype='O')
        self.embedded_models = (None, None, None, None)
        for a in xrange(num_actions):
            self.embedded_models = self.model.update_embedded_model(a, *self.embedded_models)
            Kab = self.embedded_models[0]
            self.alphas[a] = np.zeros((self.H, Kab[a,0].shape[0]))
            self.betas[a] = np.zeros((self.H, Kab[0,a].shape[1]))
    
    def step(self, x_t, a_t, r_t, x_tp1):
        if not x_t is None:
            # update models only if 'update_model' is True, if the compressed model
            # does an update, update the embedded models too.
            if self.update_models and self.model.add_samples_to_action(self.phi(x_t), self.phi(x_tp1), np.array(r_t), a_t):
                 self.embedded_models = self.model.update_embedded_model(a_t, *self.embedded_models)
                 Kab = self.embedded_models[0]
                 self.alphas[a_t] = np.zeros((self.H, Kab[a_t,0].shape[0]))
                 self.betas[a_t] = np.zeros((self.H, Kab[0,a_t].shape[1]))
        self.x_t = x_tp1
        self.plan[:-1,:] = self.plan[1:,:]
        self.plan[-1,:] = 1.0/self.plan.shape[1]
        self.plan = self.plan*0.5 + 0.5/self.plan.shape[1]
        self.improve_plan()
        
    
    def get_action(self, t = 0):
        return np.argmax(self.plan[t])
        
    
    def improve_plan(self):
        alphas = self.alphas
        betas = self.betas
        Kab, Da, wa, Va, Ua = self.embedded_models
        H = self.H
        k = alphas.shape[0]
        for a in xrange(k):
            alphas[a][H-1,:] = wa[a]*Da[a] # + 0 if self.use_valuefn else self.theta # assuming terminal reward is zero if not using value function
            betas[a][0,:] = Va[a].T.dot(self.phi(self.x_t))
            
            
        self.plan, value = self.backward_step(alphas, betas, self.plan, improve = False)
        diff = np.Infinity
        
        while np.abs(diff) > np.abs(0.001*value):
            old_plan = self.plan.copy()
            self.plan, old_val = self.forward_step(alphas, betas, self.plan, improve = True)
            self.plan, new_val = self.backward_step(alphas, betas, self.plan, improve = True)
            diff = new_val - old_val
            print new_val
            if np.allclose(old_plan, self.plan, atol = 0.01):
                break
        
        

    def backward_step(self, alphas, betas, plan, improve):
        k = alphas.shape[0]
        H = self.H
        Kab, Da, wa, Va = self.embedded_models
        
        if improve:
            new_plan = plan.copy()
        else:
            new_plan = plan
            
        gamma = self.blend_coeff
        
        vals = np.zeros(k)
        if improve:
            for a in xrange(k):
                vals[a] = alphas[a][H-1,:].dot(betas[a][H-1,:])
            a_best = np.argmax(vals)
            new_plan[H-1,:] *= gamma
            new_plan[H-1,a_best] += (1-gamma)
            
        for t in xrange(H-2, -1, -1):
            for a in xrange(k):
                alphas[a][t,:] = sum( [ new_plan[t+1, b] * alphas[b][t+1,:].dot(Kab[b,a]) * Da[a] + wa[a]*Da[a] for b in xrange(k)])
                vals[a] = alphas[a][t,:].dot(betas[a][t,:])
            
            if improve:
                a_best = np.argmax(vals)  
                new_plan[t,:] *= gamma
                new_plan[t,a_best] += (1-gamma)
                
        new_val = vals.dot(new_plan[0])
        
        return new_plan, new_val
        
    def forward_step(self, alphas, betas, plan, improve):
        k = alphas.shape[0]
        H = self.H
        Kab, Da, wa, Va = self.embedded_models
        
        if improve:
            new_plan = plan.copy()
        else:
            new_plan = plan
            
        gamma = self.blend_coeff
        
        vals = np.zeros(k)
        for a in xrange(k):
                vals[a] = betas[a][0,:].dot(alphas[a][0,:])
        if improve:
            a_best = np.argmax(vals)
            new_plan[0,:] *= gamma
            new_plan[0,a_best] += (1-gamma)
            
        old_val = plan[0].dot(vals)
            
        for t in xrange(1, H):
            for a in xrange(k):
                betas[a][t,:] = sum( [new_plan[t-1, b] * Kab[a,b].dot(Da[b]*betas[b][t-1,:]) for b in xrange(k)])
                vals[a] = betas[a][t,:].dot(alphas[a][t,:])
                
            if improve:
                a_best = np.argmax(vals)  
                new_plan[t,:] *= gamma
                new_plan[t,a_best] += (1-gamma)
                
        return new_plan, old_val
                
class SingleEmbeddedAgent(PlanningAgent):
    def __init__(self,
                 plan_horizon,
                 dim, 
                 X_t,
                 A_t, 
                 X_tp1, 
                 R_t,
                 X_term,
                 A_term,
                 R_term,
                 max_rank,
                 blend_coeff,
                 phi,
                 action_kernels,
                 actions,
                 learning_rate = 0.001,
                 discount = 0.999,
                 lamb = 0.2,
                 update_models = False,
                 use_valuefn = False):
        self.phi = phi
        self.update_models = update_models
        self.blend_coeff = blend_coeff
        self.H = plan_horizon
        self.actions = actions
        self.discount = discount
        self.max_rank = max_rank
        self.learning_rate = learning_rate

        self.use_valuefn = use_valuefn
        if use_valuefn:
            self.theta = np.zeros(dim)
        
        phi_t = phi(np.vstack((X_t, X_term)))
        phi_tp1 = np.vstack((phi(X_tp1), np.zeros((X_term.shape[0], phi_t.shape[1]))))
        R_t = np.hstack((R_t, R_term))
        A_t = np.vstack((A_t, A_term))

        self.model = SingleCompressModel(dim = dim,
                                max_rank = max_rank,
                                action_dim = actions[0].shape[0],
                                X_t = phi_t,
                                X_tp1 = phi_tp1,
                                R_t = R_t,
                                A_t = A_t,
                                lamb = lamb,
                                keep_all_samples = False)
                                
        k = len(actions)                       
        if self.use_valuefn:
            k = k+1

        self.plan = np.ones((self.H, k))/k
        
        if self.use_valuefn:
            self.plan[0,:] = 1.0/(k-1)
            self.plan[0,-1] = 0
                    
        self.action_kernels = action_kernels
        self.alphas = np.empty(k, dtype='O')
        self.betas = np.empty(k, dtype='O')
        self.embedded_models = (None, None,None, None, None)
        Kab, Da, wa, Vas, Uas = self.embedded_models
        self.embedded_models = self.model.generate_embedded_model(
            action_kernels, 
            Kab, 
            Da, 
            wa, 
            Vas, 
            Uas, 
            max_rank = self.max_rank,
            value_fn = self.theta if self.use_valuefn else None)

        for a in xrange(k):
            Kab = self.embedded_models[0]
            self.alphas[a] = np.zeros((self.H, Kab[a,0].shape[0]))
            self.betas[a] = np.zeros((self.H, Kab[0,a].shape[1]))
            
    def reset(self):
        self.plan = np.ones((self.H, k))/k
        
        if self.use_valuefn:
            self.plan[0,:] = 1.0/(k-1)
            self.plan[0,-1] = 0
    
    def step(self, x_t, a_t, r_t, x_tp1):
        if not x_t is None:
            # update models only if 'update_model' is True, if the compressed model
            # does an update, update the embedded models too.
            if self.update_models and self.model.add_samples(self.phi(x_t), a_t, np.array(r_t), self.phi(x_tp1)) :
                Kab, Da, wa, Vas, Uas = self.embedded_models
                self.embedded_models = self.model.generate_embedded_model(
                    self.action_kernels, 
                    Kab, 
                    Da, 
                    wa, 
                    Vas, 
                    Uas, 
                    max_rank = self.max_rank,
                    value_fn = self.theta if self.use_valuefn else None)
                for a in xrange(self.alphas.shape[0]):
                    Kab = self.embedded_models[0]
                    self.alphas[a] = np.zeros((self.H, Kab[a,0].shape[0]))
                    self.betas[a] = np.zeros((self.H, Kab[0,a].shape[1]))

        
        
        # update value function if it is currently being used
        if x_t is not None and self.use_valuefn:
            phi_t = self.phi(x_t)
            V_t = self.theta.dot(phi_t)
            
            delta =  self.plan_val - V_t
            self.theta += self.learning_rate * delta * phi_t
            
            V_tp1 = self.discount*self.phi(x_tp1).dot(self.theta) if x_tp1 is not None else 0
            delta = r_t + V_tp1 - V_t
            self.theta += self.learning_rate * delta * phi_t
            
        # plan for next action
        if x_tp1 is not None:
            self.x_t = x_tp1.copy()
            self.plan[:-1,:] = self.plan[1:,:]
            self.plan[-1,:] = 1.0/self.plan.shape[1]
            self.plan = self.plan*0.5 + 0.5/self.plan.shape[1]
            if self.use_valuefn:
                self.plan[0,:-1] += self.plan[0,-1]/(self.plan.shape[1]-1)
                self.plan[0,-1] = 0.0
                
            self.improve_plan()
            
            if self.use_valuefn:
                self.plan[0,:-1] += self.plan[0,-1]/(self.plan.shape[1]-1)
                self.plan[0,-1] = 0.0
    
    def get_action(self, t = 0):
        return self.actions[np.argmax(self.plan[t])]
        
    
    def improve_plan(self):
        alphas = self.alphas
        betas = self.betas
        Kab, Da, wa, Va, Ua = self.embedded_models
        H = self.H
        k = alphas.shape[0]
        for a in xrange(k):
            alphas[a][H-1,:] = wa[a]*Da[a] + ((self.discount * (wa[-1]*Da[-1]).dot(Kab[-1,a]))*Da[a] if self.use_valuefn else 0 )# assuming terminal reward is zero if not using value function
            betas[a][0,:] = Va[a].T.dot(self.phi(self.x_t))
            
            
        self.plan, value = self.backward_step(alphas, betas, self.plan, improve = False)
        diff = np.Infinity
        
        while np.abs(diff) > np.abs(0.001*value):
            old_plan = self.plan.copy()
            self.plan, old_val = self.forward_step(alphas, betas, self.plan, improve = True)
            self.plan, new_val = self.backward_step(alphas, betas, self.plan, improve = True)
            diff = new_val - old_val
            print new_val
            if np.allclose(old_plan, self.plan, atol = 0.01):
                break
        self.plan_val = new_val
        
        

    def backward_step(self, alphas, betas, plan, improve):
        k = alphas.shape[0]
        H = self.H
        Kab, Da, wa, Va, Ua = self.embedded_models
        
        if improve:
            new_plan = plan.copy()
        else:
            new_plan = plan
            
        gamma = self.blend_coeff
        
        vals = np.zeros(k)
        if improve:
            for a in xrange(k):
                vals[a] = alphas[a][H-1,:].dot(betas[a][H-1,:])
            a_best = np.argwhere(vals == np.amax(vals)).flatten()
            new_plan[H-1,:] *= gamma
            new_plan[H-1,a_best] += (1-gamma)/a_best.shape[0]
            
        for t in xrange(H-2, -1, -1):
            for a in xrange(k):
                alphas[a][t,:] = sum( [ self.discount * new_plan[t+1, b] * alphas[b][t+1,:].dot(Kab[b,a]) * Da[a] + wa[a]*Da[a] for b in xrange(k)])
                vals[a] = alphas[a][t,:].dot(betas[a][t,:])
            
            if improve:
                a_best = np.argwhere(vals == np.amax(vals)).flatten() 
                new_plan[t,:] *= gamma
                new_plan[t,a_best] += (1-gamma)/a_best.shape[0]
                
        new_val = vals.dot(new_plan[0])
        
        return new_plan, new_val
        
    def forward_step(self, alphas, betas, plan, improve):
        k = alphas.shape[0]
        H = self.H
        Kab, Da, wa, Va, Ua = self.embedded_models
        
        if improve:
            new_plan = plan.copy()
        else:
            new_plan = plan
            
        gamma = self.blend_coeff
        
        vals = np.zeros(k)
        for a in xrange(k):
                vals[a] = betas[a][0,:].dot(alphas[a][0,:])
        if improve:
            a_best = np.argwhere(vals == np.amax(vals)).flatten()
            new_plan[0,:] *= gamma
            new_plan[0,a_best] += (1-gamma)/a_best.shape[0]
            
        old_val = plan[0].dot(vals)
            
        for t in xrange(1, H):
            for a in xrange(k):
                betas[a][t,:] = sum( [new_plan[t-1, b] * Kab[a,b].dot(Da[b]*betas[b][t-1,:]) for b in xrange(k)])
                vals[a] = betas[a][t,:].dot(alphas[a][t,:])
                
            if improve:
                a_best = np.argwhere(vals == np.amax(vals)).flatten()  
                new_plan[t,:] *= gamma
                new_plan[t,a_best] += (1-gamma)/a_best.shape[0]
                
        return new_plan, old_val
        
class DebugEmbeddedAgent(PlanningAgent):
    def __init__(self,
                 plan_horizon,
                 dim, 
                 X_t,
                 A_t, 
                 X_tp1, 
                 R_t,
                 X_term,
                 A_term,
                 R_term,
                 max_rank,
                 blend_coeff,
                 phi,
                 action_kernels,
                 actions,
                 lamb = 0.2,
                 learning_rate = 0.001,
                 discount = 0.999,
                 update_models = False,
                 use_valuefn = False):
        self.phi = phi
        self.update_models = update_models
        self.blend_coeff = blend_coeff
        self.H = plan_horizon
        self.actions = actions
        self.discount = discount
        self.max_rank = max_rank
        self.learning_rate = learning_rate

        self.use_valuefn = use_valuefn
        if use_valuefn:
            self.theta = np.zeros(dim)
        
        phi_t = phi(np.vstack((X_t, X_term)))
        phi_tp1 = np.vstack((phi(X_tp1), np.zeros((X_term.shape[0], phi_t.shape[1]))))
        R_t = np.hstack((R_t, R_term))
        A_t = np.vstack((A_t, A_term))

        self.model = SingleCompressModel(dim = dim,
                                max_rank = max_rank,
                                action_dim = actions[0].shape[0],
                                X_t = phi_t,
                                X_tp1 = phi_tp1,
                                R_t = R_t,
                                A_t = A_t,
                                lamb = lamb,
                                keep_all_samples = False)
                                
        k = len(actions)                       
        if self.use_valuefn:
            k = k+1

        self.plan = np.ones((self.H, k))/k
        
        if self.use_valuefn:
            self.plan[0,:] = 1.0/(k-1)
            self.plan[0,-1] = 0
                    
        self.action_kernels = action_kernels
        self.alphas = None
        self.betas = None
#        self.embedded_models = (None, None,None, None, None)
#        Kab, Da, wa, Vas, Uas = self.embedded_models
#        self.embedded_models = self.model.generate_embedded_model(
#            action_kernels, 
#            Kab, 
#            Da, 
#            wa, 
#            Vas, 
#            Uas, 
#            max_rank = self.max_rank,
#            value_fn = self.theta if self.use_valuefn else None)
        Ma = [ self.model.get_actions_model(a_k) for a_k in action_kernels] 
        Ua, Sa, Va, impor_ratio_a = zip(*Ma)
        
        self.embedded_models = create_old_embed(zip(Ua, Sa, Va), impor_ratio_a, phi_tp1, R_t, k, lamb, phi)

#        for a in xrange(k):
#            Kab = self.embedded_models[0]
#            self.alphas[a] = np.zeros((self.H, Kab[a,0].shape[0]))
#            self.betas[a] = np.zeros((self.H, Kab[0,a].shape[1]))
#    
    def step(self, x_t, a_t, r_t, x_tp1):
        if not x_t is None:
            # update models only if 'update_model' is True, if the compressed model
            # does an update, update the embedded models too.
            if self.update_models and self.model.add_samples(self.phi(x_t), a_t, np.array(r_t), self.phi(x_tp1)) :
                Kab, Da, wa, Vas, Uas = self.embedded_models
                self.embedded_models = self.model.generate_embedded_model(
                    self.action_kernels, 
                    Kab, 
                    Da, 
                    wa, 
                    Vas, 
                    Uas, 
                    max_rank = self.max_rank,
                    value_fn = self.theta if self.use_valuefn else None)
                for a in xrange(self.alphas.shape[0]):
                    Kab = self.embedded_models[0]
                    self.alphas[a] = np.zeros((self.H, Kab[a,0].shape[0]))
                    self.betas[a] = np.zeros((self.H, Kab[0,a].shape[1]))

        self.x_t = x_tp1.copy()
        self.plan[:-1,:] = self.plan[1:,:]
        self.plan[-1,:] = 1.0/self.plan.shape[1]
        self.plan = self.plan*0.5 + 0.5/self.plan.shape[1]
        if self.use_valuefn:
            self.plan[0,:-1] += self.plan[0,-1]/(self.plan.shape[1]-1)
            self.plan[0,-1] = 0.0
            
        self.improve_plan()
        
        if self.use_valuefn:
            self.plan[0,:-1] += self.plan[0,-1]/(self.plan.shape[1]-1)
            self.plan[0,-1] = 0.0
        

        if x_t is not None and self.use_valuefn:
            phi_t = self.phi(x_t)
            V_t = self.theta.dot(phi_t)
            
            delta =  self.plan_val - V_t
            self.theta += self.learning_rate * delta * phi_t
            
            V_tp1 = self.discount*self.phi(x_tp1).dot(self.theta) if x_tp1 is not None else 0
            delta = r_t + V_tp1 - V_t
            self.theta += self.learning_rate * delta * phi_t
    
    def get_action(self, t = 0):
        return self.actions[np.argmax(self.plan[t])]
        
    
    def improve_plan(self):
#        alphas = self.alphas
#        betas = self.betas
#        Kab, Da, wa, Va, Ua = self.embedded_models
#        H = self.H
#        k = alphas.shape[0]
#        for a in xrange(k):
#            alphas[a][H-1,:] = wa[a]*Da[a] + ((self.discount * (wa[-1]*Da[-1]).dot(Kab[-1,a]))*Da[a] if self.use_valuefn else 0 )# assuming terminal reward is zero if not using value function
#            betas[a][0,:] = Va[a].T.dot(self.phi(self.x_t))
#            
#            
#        self.plan, value = self.backward_step(alphas, betas, self.plan, improve = False)
#        diff = np.Infinity
#        
#        while np.abs(diff) > np.abs(0.001*value):
#            old_plan = self.plan.copy()
#            self.plan, old_val = self.forward_step(alphas, betas, self.plan, improve = True)
#            self.plan, new_val = self.backward_step(alphas, betas, self.plan, improve = False)
#            diff = new_val - old_val
#            print new_val
#            if np.allclose(old_plan, self.plan, atol = 0.01):
#                break
#        self.plan_val = new_val
#        
        self.plan, self.alphas, self.betas = find_stoc_plan(self.x_t, 
                                               self.H, 
                                               3, 
                                               self.embedded_models, 
                                               self.blend_coeff,
                                               approx_np = True, 
                                               forward = True, 
                                               sparse = False, 
                                               plan = self.plan, 
                                               alphas = self.alphas, 
                                               betas = self.betas)
        

    def backward_step(self, alphas, betas, plan, improve):
        k = alphas.shape[0]
        H = self.H
        Kab, Da, wa, Va, Ua = self.embedded_models
        
        if improve:
            new_plan = plan.copy()
        else:
            new_plan = plan
            
        gamma = self.blend_coeff
        
        vals = np.zeros(k)
        if improve:
            for a in xrange(k):
                vals[a] = alphas[a][H-1,:].dot(betas[a][H-1,:])
            a_best = np.argmax(vals)
            new_plan[H-1,:] *= gamma
            new_plan[H-1,a_best] += (1-gamma)
            
        for t in xrange(H-2, -1, -1):
            for a in xrange(k):
                alphas[a][t,:] = sum( [ self.discount * new_plan[t+1, b] * alphas[b][t+1,:].dot(Kab[b,a]) * Da[a] + wa[a]*Da[a] for b in xrange(k)])
                vals[a] = alphas[a][t,:].dot(betas[a][t,:])
            
            if improve:
                a_best = np.argmax(vals)  
                new_plan[t,:] *= gamma
                new_plan[t,a_best] += (1-gamma)
                
        new_val = vals.dot(new_plan[0])
        
        return new_plan, new_val
        
    def forward_step(self, alphas, betas, plan, improve):
        k = alphas.shape[0]
        H = self.H
        Kab, Da, wa, Va, Ua = self.embedded_models
        
        if improve:
            new_plan = plan.copy()
        else:
            new_plan = plan
            
        gamma = self.blend_coeff
        
        vals = np.zeros(k)
        for a in xrange(k):
                vals[a] = betas[a][0,:].dot(alphas[a][0,:])
        if improve:
            a_best = np.argmax(vals)
            new_plan[0,:] *= gamma
            new_plan[0,a_best] += (1-gamma)
            
        old_val = plan[0].dot(vals)
            
        for t in xrange(1, H):
            for a in xrange(k):
                betas[a][t,:] = sum( [new_plan[t-1, b] * Kab[a,b].dot(Da[b]*betas[b][t-1,:]) for b in xrange(k)])
                vals[a] = betas[a][t,:].dot(alphas[a][t,:])
                
            if improve:
                a_best = np.argmax(vals)  
                new_plan[t,:] *= gamma
                new_plan[t,a_best] += (1-gamma)
                
        return new_plan, old_val
                