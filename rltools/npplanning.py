import numpy as np
import scipy

from itertools import product, izip

from scipy.linalg import lu_factor, lu_solve

import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.spatial.distance import pdist, squareform

from scikits.sparse.cholmod import cholesky
import matplotlib.pyplot as plt

import cvxpy

# A constraint Generation Approach to Learning Stable Linear Dynamical Systems
# Sajid M. Siddiqi, Bryon Boots, Geoffry J. Gordon
def compute_stable_model(X_t, X_tp1):
    
    d = X_t.shape[1]
    A = cvxpy.Variable(d,d)    
    cost = cvxpy.Minimize(cvxpy.norm( X_t * A - X_tp1, 'fro' ))
    
    F_star = np.linalg.pinv(X_t).dot(X_tp1).T
    F = F_star
    
    constraints = [cvxpy.bmat([[ np.eye(d), A], [A.T, np.eye(d)]])>> 0]
    prob = cvxpy.Problem(cost, constraints)
    v = prob.solve('SCS')
    print 'val: ', v
    F = np.array(A.value)
#    max_eig = np.abs(np.linalg.eigvals(F)).max()
#    while max_eig > 1:
#        print max_eig
#        U, S, Vt = np.linalg.svd(F, full_matrices = False)
#        u = U[:,0]
#        v = Vt[0,:]
#        constraints.append( cvxpy.vec(u[:,None]*v[None,:]).T * cvxpy.vec(A) <= 1)
#        prob = cvxpy.Problem(cost, constraints)
#        v = prob.solve( 'SCS', warm_start = True)
#        print 'val: ', v
#        F = np.array(A.value)
#        max_eig = np.abs(np.linalg.eigvals(F)).max()
        
    # missing binary search for better solution
    return F


# Aki Rahimi and Ben Recht, Random Features for Large-Scale Kernel Machines
def sample_gaussian(d, num_gauss = 1, scaling = None):
    w = np.random.normal(0.0, 1.0, size = (d, num_gauss))
    if scaling is not None:
        w = w/scaling[:,None]
        
    #w = np.vstack((w, np.random.uniform(-np.pi, np.pi, size = (1, num_gauss))))
    w = np.vstack((w, np.zeros((1, num_gauss))))
    return w


def compress_phi_model(X, Xp, R, k, lamb):
    U,S,Vt = np.linalg.svd(X, full_matrices = False)
#    Up, Sp, Vpt = np.linalg.svd(Xp, full_matrices = False)
    k = min(k, X.shape[0])
    k2 = min(k, Xp.shape[0])
    
    r = Vt[:k,:].T.dot((S[:k]/(S[:k]**2 + lamb))* (U[:,:k].T.dot(R)))
    
    
    Vpt = U[:,:k].T.dot(Xp)
    S = np.diag(S[:k]/(S[:k]**2 + lamb))
    return (Vpt.T, S.T, Vt[:k,:]), r
    
    
#    S = np.diag(S[:k]/(S[:k]**2 + lamb)).dot(U[:,:k].T.dot(Up[:,:k2].dot(np.diag(Sp[:k2]))))
#    return (Vpt[:k2,:].T, S.T, Vt[:k,:]), r
#    S = np.diag(S/(S**2 + lamb)).dot(U.T.dot(Up.dot(np.diag(Sp))))
#    return Vpt.T, S.T, Vt
#    
    
def build_approx_gauss_models(scale, 
                              trans_samples, 
                              ter_samples,
                              phi,
                              num_gauss = 200,
                              k = 200,
                              lamb = 0.15):
                        
    
    Xa, Ra, Xpa = zip(*trans_samples)
    Xa_term, Ra_term = zip(*ter_samples)

    Fa = []
    ra = []
    
    for X, R, Xp, X_term, R_term in zip(Xa, Ra, Xpa, Xa_term, Ra_term):
        if X_term.size > 0:
            phis = phi(np.vstack((X, X_term)))
        else:
            phis = phi(X)
        
        phip = phi(Xp)
        if X_term.size > 0:
            phip =np.vstack((phip, np.zeros((X_term.shape[0], phip.shape[1]))))        
        
        R = np.hstack((R,R_term))  
        
        F, r = compress_phi_model(phis, phip, R, k, lamb = lamb)
        ra.append(r)
        #Fa.append(np.linalg.lstsq(phis, phip, rcond = .01)[0].T)
        
        Fa.append(F)
        
        
        #F = np.linalg.svd( np.linalg.lstsq(phis, phip, rcond = .01)[0].T, full_matrices=0)
       # F = (F[0][:,:k], F[1][:k], F[2][:k, :])
        #Fa.append(F)
        
        
        
        #Fa.append(np.linalg.lstsq(phis.T.dot(phis) + 0.15 * np.eye(phis.shape[1]), phis.T.dot(phip), rcond = .0001)[0].T)
        #Fa.append(compute_stable_model(phis, phip))
        
    return np.array(Fa), np.array(ra), phi
    #return Fa, np.array(ra), w
    
    
                                          
    


def build_np_models(kernel, trans_samples, ter_samples, ter_rew_samples, lamb):
    Xa, Ra, Xpa = zip(*trans_samples)
    Xa_term, Ra_term = zip(*ter_samples)
    Xa = [ np.vstack((xa, xa_term)) if xa_term.size > 0 else xa for xa, xa_term in izip(Xa, Xa_term) ]
    Ra = [ np.hstack((ra, ra_term)) if ra_term.size > 0 else ra for ra, ra_term in izip(Ra, Ra_term) ] 
    
    k = len(trans_samples) 
    
    # build the K_a,b matrices
    Kab = dict()
    for a,b in product(xrange(k), xrange(k)):
        if Xa_term[b].size > 0:
            Kab[(a,b)] = np.hstack((kernel(Xa[a], Xpa[b]), 
                                    np.zeros((Xa[a].shape[0], Xa_term[b].shape[0]))))
        else:
            Kab[(a,b)] = kernel(Xa[a], Xpa[b])
        
    
    # build the K_a, D_a matrices
    Ka = [kernel(Xa[i], Xa[i])  for i in xrange(k)]
    Dainv = [Ka[i] + lamb*scipy.eye(*Ka[i].shape) for i in xrange(k)]
    Da = [lu_factor(Dainv[i], overwrite_a = False) for i in xrange(k)]
        
    # build K_ter matrix
    Kterma = [ np.hstack((kernel(ter_rew_samples[0], Xpa[i]),
                          np.zeros((ter_rew_samples[0].shape[0], Xa_term[i].shape[0])))) if Xa_term[i].size > 0
                else kernel(ter_rew_samples[0], Xpa[i]) for i in xrange(k)]
    K_ter = kernel(ter_rew_samples[0], ter_rew_samples[0])
    D_ter = lu_factor(K_ter + lamb*scipy.eye(*K_ter.shape), overwrite_a = True)
    R_ter = ter_rew_samples[1]
    
    return kernel, Kab, Da, Dainv, Ra, Kterma, D_ter, R_ter, Xa
    
def sparse_build_np_models(kernel, trans_samples, ter_samples, ter_rew_samples, lamb):
    Xa, Ra, Xpa = zip(*trans_samples)
    Xa_term, Ra_term = zip(*ter_samples)
    Xa = [ np.vstack((xa, xa_term)) if xa_term.size > 0 else xa for xa, xa_term in izip(Xa, Xa_term) ]
    Ra = [ np.hstack((ra, ra_term)) if ra_term.size > 0 else ra for ra, ra_term in izip(Ra, Ra_term) ] 
    
    k = len(trans_samples) 
    
    # build the K_a,b matrices
    Kab = dict()
    KabT = dict()
    for a,b in product(xrange(k), xrange(k)):
        if Xa_term[b].size > 0:
            Kab[(a,b)] = np.hstack((kernel(Xa[a], Xpa[b]), 
                                    np.zeros((Xa[a].shape[0], Xa_term[b].shape[0]))))
        else:
            Kab[(a,b)] = kernel(Xa[a], Xpa[b])
        Kab[(a,b)] = csr_matrix(Kab[(a,b)] * (np.abs(Kab[(a,b)]) > 1e-3))
        KabT[(a,b)] = Kab[(a,b)].T.tocsr()
    
    # build the K_a, D_a matrices
    Ka = [kernel(Xa[i], Xa[i])  for i in xrange(k)]
    Dainv = [csc_matrix(Ka[i] * (np.abs(Ka[i]) > 1e-3)) + lamb*scipy.sparse.eye(*Ka[i].shape) for i in xrange(k)]
#    print np.linalg.matrix_rank(Dainv[2].toarray()), Dainv[2].shape, np.linalg.cond(Dainv[2].toarray())
#    print np.linalg.eig(Dainv[2].toarray())[0]
#    print [np.linalg.eig(Dainv[i].toarray())[0].min() for i in xrange(3)]
#    plt.spy(Dainv[2].toarray())
#    plt.show()  
#    print Dainv[2].shape

    index = (squareform(pdist(Xa[2])) == 0.0).nonzero()    
    
#    print (squareform(pdist(Xa[2])) == 0.0).nonzero()
#    print Xa[2][index[0][:5],:]
#    print Xa[2][index[1][:5],:]
#    
#    splu(Dainv[0])    
#    cholesky(Dainv[0])
#    splu(Dainv[1])
#    cholesky(Dainv[1])
#    splu(Dainv[2])
    cholesky(Dainv[2])
        
    
    Da= [cholesky(Dainv[i]) for i in xrange(k)]
#    Da = [splu(Dainv[i]) for i in xrange(k)]
        
    # build K_ter matrix
    Kterma = [ np.hstack((kernel(ter_rew_samples[0], Xpa[i]),
                          np.zeros((ter_rew_samples[0].shape[0], Xa_term[i].shape[0])))) if Xa_term[i].size > 0
                else kernel(ter_rew_samples[0], Xpa[i]) for i in xrange(k)]
    K_ter = kernel(ter_rew_samples[0], ter_rew_samples[0])
    D_ter = cholesky(csc_matrix(K_ter*(np.abs(K_ter) > 1e-3)) + lamb*scipy.sparse.eye(*K_ter.shape))
#    D_ter = splu(csc_matrix(K_ter*(np.abs(K_ter) > 1e-3)) + lamb*scipy.sparse.eye(*K_ter.shape))
    R_ter = ter_rew_samples[1]
    
    return kernel, Kab, KabT, Da, Dainv, Ra, Kterma, D_ter, R_ter, Xa
    
    
def approx_np_improve(plan, 
                      gamma,
                      Fa,
                      ra,
                      phi,
                      x_1,
                      alphas = None,
                      betas = None,
                      forward = True):
    H = plan.shape[0]                     
    alpha = np.zeros((H+1, Fa.shape[1]))
    beta = np.zeros((H, Fa.shape[1]))
    new_plan = plan.copy()
    
    beta[0,:] = phi(x_1).squeeze()
    discount = 1.0
    if forward:
        for t in xrange(H-1, -1, -1):
            alpha[t,:] = (alpha[t+1,:]).dot(np.tensordot(Fa, plan[t,:], axes=(0,0)))*discount + np.tensordot(ra, plan[t,:], axes=(0,0))
            
        old_val = alpha[0].dot(beta[0])
        
        for t in xrange(0, H):
            if t > 0:
                beta[t,:] = np.tensordot(Fa, new_plan[t-1,:], axes=(0,0)).dot(beta[t-1,:])
            va = np.tensordot(Fa, beta[t], axes=(2,0)).dot(alpha[t+1])*discount + ra.dot(beta[t])
            
            a_best = np.argmax(va)
            new_plan[t,:] *= gamma
            new_plan[t,a_best] += (1-gamma)
    else:
        raise Exception('Not Implemented')
        
    return new_plan, old_val, alpha, beta
        
def approx_np_improve_compressed(plan, 
                      gamma,
                      Fa,
                      ra,
                      phi,
                      x_1,
                      alphas = None,
                      betas = None,
                      forward = True):
    H = plan.shape[0]                     
    alpha = np.zeros((H+1, Fa[0][0].shape[0]))
    beta = np.zeros((H, Fa[0][0].shape[0]))
    new_plan = plan.copy()
    
    beta[0,:] = phi(x_1).squeeze()
    discount = 1.0
    if forward:
        for t in xrange(H-1, -1, -1):
            U, S, Vt = Fa[0]
            a = ((alpha[t+1,:].dot(U)).dot(S)).dot(Vt) * plan[t,0]
            for F, p in izip(Fa[1:], plan[t,1:]):
                U, S, Vt = F
                a += ((alpha[t+1,:].dot(U)).dot(S)).dot(Vt) * p
                
            alpha[t,:] = a*discount + np.tensordot(ra, plan[t,:], axes=(0,0))
            
        old_val = alpha[0].dot(beta[0])
        
        for t in xrange(0, H):
            if t > 0:
                U, S, Vt = Fa[0]
                b = U.dot(S.dot( Vt.dot(beta[t-1,:]))) * new_plan[t-1,0]
                for F, p in izip(Fa[1:], new_plan[t-1,1:]):
                    U, S, Vt = F
                    b += U.dot(S.dot(Vt.dot(beta[t-1,:]))) * p
                beta[t,:] = b
            va = [ alpha[t+1].dot(F[0].dot(F[1].dot(F[2].dot(beta[t,:]))))*discount + r.dot(beta[t]) for r, F in izip(ra, Fa)]         
            a_best = np.argmax(va)
            new_plan[t,:] *= gamma
            new_plan[t,a_best] += (1-gamma)
    else:
        raise Exception('Not Implemented')
        
#    
#    for t in xrange(H-1, -1, -1):
#            alpha[t,:] = (alpha[t+1,:]).dot(np.tensordot(Fa, new_plan[t,:], axes=(0,0))) + np.tensordot(ra, new_plan[t,:], axes=(0,0))
    #print alpha[0].dot(beta[0]), old_val, old_val <= alpha[0].dot(beta[0])
    
    return new_plan, old_val, alpha, beta
    
def convert_compressed_to_embed(Fa, ra, phi):
    k = len(Fa)
    Ua = [ Fa[a][0] for a in xrange(k)]
    Va = [ Fa[a][1].dot(Fa[a][2]) for a in xrange(k)]

    Kab = np.empty(shape = (k,k), dtype = 'O')
    thetas = np.empty(shape = (k,k), dtype = 'O') 
    for a in xrange(k):
        for b in xrange(k):
            Kab[a,b] = Va[a].dot(Ua[b])
            thetas[a,b] = ra[a].dot(Ua[b])
            
    return Va, Kab, thetas, ra, phi
            
    
def approx_embed_improve(plan,
                         gamma,
                         Va,
                         Kab,
                         thetas,
                         ra,
                         phi,
                         x_1,
                         alphas = None,
                         betas = None,
                         forward = True):
                             

    H = plan.shape[0] - 1
    k = plan.shape[1]
    if alphas is None:
        alphas = np.empty(k, dtype='O')
        for a in xrange(k):                     
            alphas[a] = np.zeros((H+1, Kab[a,0].shape[0]))
    if betas is None:
        betas = np.empty(k, dtype='O')
        for a in xrange(k):                     
            betas[a] = np.zeros((H, Kab[0,a].shape[1]))
    new_plan = plan.copy()
    
    phi_0 = phi(x_1)
    for a in xrange(k):
        betas[a][0,:] = Va[a].dot(phi_0)
        alphas[a][H,:] = sum( [plan[H][b] * thetas[b,a] for b in xrange(k)])
    
    if forward:
        for t in xrange(H, 0, -1):
            for a in xrange(k):
                alphas[a][ t-1,:] = sum( [plan[t-1][b] * (alphas[b][t].dot(Kab[b,a]) + thetas[b,a]) for b in xrange(k)])
                
        for t in xrange(0, H):
            vals = np.zeros(k)
            if t > 0:
                for a in xrange(k):
                    betas[a][t,:] = sum( [new_plan[t-1][b] * Kab[a,b].dot(betas[b][t-1]) for b in xrange(k)])
                for a in xrange(k):
                    vals[a] = alphas[a][ t+1].dot(betas[a][t,:]) + sum( [thetas[a,b].dot(betas[b][t-1]) * new_plan[t-1][b] for b in xrange(k)])
            else:
                for a in xrange(k):
                    vals[a] = alphas[a][ t+1].dot(betas[a][t,:]) + ra[a].dot(phi_0)       
                old_val = plan[0].dot(vals)
            a_best = np.argmax(vals)
            new_plan[t,:] *= gamma
            new_plan[t,a_best] += (1-gamma)
            
        
    else:
        raise Exception('Not implemented')

    return new_plan, old_val, alphas, betas
    
@profile        
def non_param_improve(plan, 
                      gamma, 
                      kernel, 
                      Kab, 
                      Da, 
                      Dainv,
                      Ra, 
                      Kterma, 
                      Dterm, 
                      Rterm, 
                      Xa, 
                      x_1,
                      alphas = None,
                      betas = None,
                      forward = True):
    H = plan.shape[0]
    new_plan = plan * gamma   
    k = len(Dainv)
    if alphas is None:
        alphas = [ np.zeros((H, Xa[a].shape[0])) for a in xrange(k)]

    if betas is None:    
        betas = [ np.zeros((H, Xa[a].shape[0])) for a in xrange(k)]
    
    for a in xrange(k):
        alphas[a][H-1, :] = lu_solve( Da[a], Ra[a] + Kterma[a].T.dot(lu_solve( Dterm, Rterm, trans = 1)), trans = 1)
        betas[a][0,:] = lu_solve(Da[a], kernel(Xa[a], x_1))
        
    
    va = np.empty(k)
    if forward:
        for t in xrange(H-2, -1, -1):
            for a in xrange(k):
                a_prime = sum([ plan[t+1,b] * Kab[b, a].T.dot(alphas[b][t+1,:]) for b in xrange(k)])
                alphas[a][t,:] = lu_solve(Da[a], Ra[a] + a_prime, trans = 1)

        for a in xrange(k):
            va[a] = alphas[a][0,:].dot(Dainv[a].dot(betas[a][0,:]))
        old_val = plan[0,:].dot(va)    
        
        for t in xrange(0, H):
            for a in xrange(k):
                if t > 0:
                    b_prime = sum( [Kab[a,b].dot(betas[b][t-1,:]) * new_plan[t-1,b] for b in xrange(k)])
                    betas[a][t] = lu_solve(Da[a], b_prime, trans = 0)
                va[a] = alphas[a][t,:].dot(Dainv[a].dot(betas[a][t,:]))
                
            a_best = np.argmax(va)
            new_plan[t,a_best] += (1-gamma)
    else:
        for t in xrange(1, H-1):
            for a in xrange(k):
                b_prime = sum( [Kab[a,b].dot(betas[b][t-1,:]) * plan[t-1,b] for b in xrange(k)])
                betas[a][t] = lu_solve(Da[a], b_prime, trans = 0)
                
        for t in xrange(H-1,-1, -1):
            for a in xrange(k):
                if t< H-1:
                    a_prime = sum([ new_plan[t+1,b] * Kab[b, a].T.dot(alphas[b][t+1,:]) for b in xrange(k)])
                    alphas[a][t,:] = lu_solve(Da[a], Ra[a] + a_prime, trans = 1)
                va[a] = alphas[a][t,:].dot(Dainv[a].dot(betas[a][t,:]))
                
            a_best = np.argmax(va)
            new_plan[t,a_best] += (1-gamma)
            
        
        for a in xrange(k):
            va[a] = alphas[a][0,:].dot(Dainv[a].dot(betas[a][0,:]))
        old_val = new_plan[0,:].dot(va) 
    
    return new_plan, old_val, alphas, betas
    
@profile        
def sparse_non_param_improve(plan, 
                              gamma, 
                              kernel, 
                              Kab,
                              KabT,
                              Da, 
                              Dainv,
                              Ra, 
                              Kterma, 
                              Dterm, 
                              Rterm, 
                              Xa, 
                              x_1,
                              alphas = None,
                              betas = None,
                              forward = True):
    H = plan.shape[0]
    new_plan = plan * gamma   
    k = len(Dainv)
    if alphas is None:
        alphas = [ np.zeros((H, Xa[a].shape[0])) for a in xrange(k)]

    if betas is None:    
        betas = [ np.zeros((H, Xa[a].shape[0])) for a in xrange(k)]
    
    for a in xrange(k):
#        alphas[a][H-1, :] = Da[a].solve( Ra[a] + Kterma[a].T.dot( Dterm.solve(Rterm, trans = 'T')), trans = 'T')
#        betas[a][0,:] = Da[a].solve( kernel(Xa[a], x_1).squeeze(), trans='N')
        alphas[a][H-1, :] = Da[a]( Ra[a] + Kterma[a].T.dot( Dterm(Rterm)))
        betas[a][0,:] = Da[a]( kernel(Xa[a], x_1).squeeze())
        
    
    va = np.empty(k)
    if forward:
        for t in xrange(H-2, -1, -1):
            for a in xrange(k):
                a_prime = sum([ plan[t+1,b] * KabT[b, a].dot(alphas[b][t+1,:]) for b in xrange(k)])
#                alphas[a][t,:] = Da[a].solve( Ra[a] + a_prime, trans = 'T')
                alphas[a][t,:] = Da[a]( Ra[a] + a_prime)

        for a in xrange(k):
            va[a] = alphas[a][0,:].dot(Dainv[a].dot(betas[a][0,:]))
        old_val = plan[0,:].dot(va)    
        
        for t in xrange(0, H):
            for a in xrange(k):
                if t > 0:
                    b_prime = sum( [Kab[a,b].dot(betas[b][t-1,:]) * new_plan[t-1,b] for b in xrange(k)])
#                    betas[a][t] = Da[a].solve( b_prime, trans = 'N')
                    betas[a][t] = Da[a]( b_prime)
                va[a] = alphas[a][t,:].dot(Dainv[a].dot(betas[a][t,:]))
                
            a_best = np.argmax(va)
            new_plan[t,a_best] += (1-gamma)
    else:
        for t in xrange(1, H-1):
            for a in xrange(k):
                b_prime = sum( [Kab[a,b].dot(betas[b][t-1,:]) * plan[t-1,b] for b in xrange(k)])
#                betas[a][t] = Da[a].solve( b_prime, trans = 'N')
                betas[a][t] = Da[a]( b_prime)
#                betas[a][t] = np.clip(betas[a][t], 0, 1)
#                betas[a][t] = betas[a][t]/ np.sum(np.abs(betas[a][t]))

        for a in xrange(k):
            va[a] = alphas[a][0,:].dot(Dainv[a].dot(betas[a][0,:]))
        old_val = plan[0,:].dot(va)    
        
        for t in xrange(H-1,-1, -1):
            for a in xrange(k):
                if t< H-1:
                    a_prime = sum([ new_plan[t+1,b] * KabT[b, a].dot(alphas[b][t+1,:]) for b in xrange(k)])
#                    alphas[a][t,:] = Da[a].solve( Ra[a] + a_prime, trans = 'T')
                    alphas[a][t,:] = Da[a]( Ra[a] + a_prime)
                va[a] = alphas[a][t,:].dot(Dainv[a].dot(betas[a][t,:]))
                
            a_best = np.argmax(va)
            new_plan[t,a_best] += (1-gamma)
    
    return new_plan, old_val, alphas, betas

def get_random_plan(T, n):
    p0 = np.abs(np.ones((T,n))/n)# + np.random.normal(loc = 0.0, scale = 0.01, size = (T, n)))
    p0 = p0/ (np.sum(p0, axis=1)[:,None])
    return p0

def check_convergence(p0, p1):
    converged = np.allclose(p0, p1)
    return converged

@profile
def find_stoc_plan(x_1, H, num_actions, models, gamma, approx_np = True, forward = True, sparse = False, plan = None, alphas = None, betas = None):
    
    
    if approx_np:
        improve_step = approx_embed_improve
        #improve_step = approx_np_improve_compressed
    else:
        if sparse:
            improve_step = sparse_non_param_improve
        else:
            improve_step = non_param_improve
    
    if plan is None:
        p0 = get_random_plan(H, num_actions)
    else:
        p0 = plan
    p1, best_val, alphas, betas = improve_step(p0, gamma, *models, x_1 = x_1, forward = forward)
    i = 0
    while not (check_convergence(p0, p1) or i>4):
        p0 = p1
        p1, val, alphas, betas = improve_step(p0, 
                                                   gamma, 
                                                   *models, 
                                                   x_1 = x_1, 
                                                   alphas = alphas, 
                                                   betas = betas,
                                                   forward = forward)
        if best_val != 0:
            norm_val = best_val
        elif val != 0:
            norm_val = val
        else:
            norm_val = 1.0
            
        if np.abs(val - best_val)/np.abs(norm_val) <1e-2:
            i += 1
        else:
            i=0
        print i, val
        best_val = val
        if np.isnan(best_val):
            break
            
    return p1, alphas, betas
        
                            

class NonParametricModel(object):
    def __init__(self, kernel, trans_samples, ter_samples, lamb):
        rew_models, trans_models, ter_model = build_np_models(kernel, 
                                                              trans_samples, 
                                                              ter_samples, 
                                                              lamb)
        
        
    