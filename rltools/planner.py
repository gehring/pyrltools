# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 12:31:35 2016

@author: cgehri
"""
import numpy as np
from rltools.model import CompressedModel


class PlanningAgent:
    def __init__(self):
        raise NotImplementedError('Should be implemented by subclass')
    
    def get_next_action(self):
        raise NotImplementedError('Should be implemented by subclass')
    
    def step(self, x_t, a_t, r_t, x_tp1):
        raise NotImplementedError('Should be implemented by subclass')
        
    
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
                 update_models = False):
        self.phi = phi
        self.update_models = update_models
        self.blend_coeff = blend_coeff
        self.H = plan_horizon
        
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
        Kab, Da, wa, Va = self.embedded_models
        H = self.H
        k = alphas.shape[0]
        for a in xrange(k):
            alphas[a][H-1,:] = wa[a]*Da[a] # assuming terminal reward is zero
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
                