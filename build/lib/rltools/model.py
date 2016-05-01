# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:16:31 2016

@author: cgehri
"""
import numpy as np

class CompressedModel:
    """

    Xa_t:   List of matrices representing start states organized row-wise. Each
            There should be one matrix for every action.
            
    Xa_tp1: List of matrices representing end states organized row-wise. Each
            There should be one matrix for every action.
            
    Notes:  ideas for avoiding keeping track of all end states.
            1. We can compress the end states to compactly represent all end 
               states while scaling more favorably
    """
    def __init__(self, 
                 dim,
                 num_actions,
                 max_rank,
                 lamb = 0.2,
                 Xa_t = None, 
                 Xa_tp1 = None, 
                 Ra = None,
                 keep_all_samples = True,
                 compress_end_states = False,
                 initialize_uniform = False):
        self.dim = dim
        self.num_actions = num_actions
        self.max_rank = max_rank
        self.lamb = lamb
        self.keep_all_samples = keep_all_samples
        self.compress_end_states = compress_end_states
            
        
        # how many samples are in the buffer waiting to be processed
        self.count = [0 for i in xrange(num_actions)]
        
        
        if initialize_uniform:
            raise NotImplementedError()
        else:
            self.ab_buffer = [np.zeros((dim, max_rank)) for i in xrange(num_actions)]
            self.Ma = [None]*num_actions
            if compress_end_states:
                self.Mpa = [None]*num_actions
            
            
        if (not compress_end_states) or keep_all_samples:
                self.Xa_tp1 = [ BufferedRowCollector(column_dim = dim, rows = None) for X_tp1 in Xa_tp1]
        if not Xa_t is None:
            self.Ra = [  BufferedRowCollector(column_dim = 1, rows = None) for R in Ra]
            for i in xrange(num_actions):
                self.add_samples_to_action(Xa_t[i], Xa_tp1[i], Ra[i], i)
                self.update_and_clear_buffer(i)
            if keep_all_samples:
                self.Xa_t = [ BufferedRowCollector(column_dim = dim, rows = X_t) for X_t in Xa_t]
                
            
                

    """
    Update the compressed matrix M with an outer product of two matrices A, B
    such that the returned compressed matrix M' = M + AB^T.
    
    CURRENTLY NO OPTIMIZATION FOR ADDING ROW OPERATION
    """
    def __update_matrix(self, M, A, B):
        if M is not None:
            U,S,V = M
            U = np.pad(U, ((0,A.shape[0] - U.shape[0]), (0,0)), mode = 'constant')
            Up = U.copy()
            Vp = V.copy()
            Sp = S.copy()
            
            # compute value for the left and right subspace
            Q_a, R_a = np.linalg.qr(A - U.dot(U.T.dot(A)), mode='reduced')
            Q_b, R_b = np.linalg.qr(B - V.dot(V.T.dot(B)), mode='reduced')
    
            Ap = np.vstack((U.T.dot(A), R_a))
            Bp = np.vstack((V.T.dot(B), R_b))
    
            numiters = 5
            success = 1
            for i in range(numiters):
                try:
                    # naive diagonalization of the center matrix (i.e., with SVD)
                    K = np.diag(np.hstack((S, np.zeros(R_a.shape[0])))) + Ap.dot(Bp.T)
                    Up, Sp, Vp = np.linalg.svd(K, full_matrices = False)
                    success = 1
                except:    
                    success = 0
                if success == 1:
                    break
    
            if success == 1:
                # update left and right singular vectors
                U = np.hstack((U, Q_a)).dot(Up)
                V = np.hstack((V, Q_b)).dot(Vp.T)
            
                self.rank = min(self.max_rank, Sp.size)
            else:
                print 'SVD failed to converge, continuing anyway ...\n'
        else:
            # initialize, assuming matrix is all zeroes
            
            # compute value for the left and right subspace
            Q_a, R_a = np.linalg.qr(A, mode='reduced')
            Q_b, R_b = np.linalg.qr(B, mode='reduced')
            
            # initial diagonalization 
            Up, Sp, Vp = np.linalg.svd(R_a.dot(R_b.T), full_matrices = False)

            # construct the singular vectors
            U = Q_a.dot(Up)
            V = Q_b.dot(Vp.T)
            self.initialized = True
            
            # Currently, we don't truncate singular values close to zero and
            # keep the full max_rank approximation. We could potentially remove
            # singular values close to zero.
            self.rank = min(self.max_rank, Sp.size)
            
        S = Sp[:self.rank]
        U = U[:,:self.rank]
        V = V[:,:self.rank]
        return U,S,V
        
    """ Update svd with a rank-k outer-product, where k = self.count
    """           
    def update_and_clear_buffer(self, action):
        if self.count[action] == 0:
            return
        B = self.ab_buffer[action]
        A = np.pad(np.eye(self.count[action]), (( (self.Ma[action][0].shape[0] if not self.Ma[action] is None else 0), 0), (0,0)), mode = 'constant') 
        
        self.Ma[action] = self.__update_matrix(self.Ma[action],A, B[:,:self.count[action]])
        B[:,:] = 0.0
        self.count[action] = 0
        self.ab_buffer[action] = B
    
    """
    Add sample to the linear expectation model of action. The sample transitions are 
    organized as follows X_t has the start statesorganized row-wise and X_tp1 
    has the end states organized row-wise. R is a vector of the reward recieved
    after each transition.
    """
    def add_samples_to_action(self, X_t, X_tp1, R, action):
        if self.compress_end_states:
            raise NotImplementedError()
            
        if X_t.ndim == 1:
            X_t = X_t.reshape((1,-1))
        if X_tp1.ndim == 1:
            X_tp1 = X_tp1.reshape((1,-1))

        self.Xa_tp1[action].add_rows(X_tp1) 
        self.Ra[action].add_rows(R.reshape((-1,1)))
        
        transX_t = X_t.T
            
        # save the current update
        B = self.ab_buffer[action]
        
        # if the whole update doesn't fit in the buffer, iterate over
        # chunks that do
        model_updated = False
        j = 0
        while j<transX_t.shape[1]:
            i = self.count[action] + transX_t.shape[1] - j
            i_clamp = min(i, self.max_rank)
            jp = j + i_clamp - self.count[action]
               
            # place sub-matrix into buffer
            mat_b = transX_t[:, j:jp]
            B[:, self.count[action]:i_clamp] = mat_b
            
            self.count[action] = i_clamp
            j = jp
            
            # if the buffer is full, update svd
            if self.count[action] >= self.max_rank:
                self.update_and_clear_buffer(action)       
                model_updated = True
        return model_updated
        
    
    """
    Add sample to the linear expectation models. The sample transitions are 
    organized as follows X_t has the start statesorganized row-wise and X_tp1 
    has the end states organized row-wise. R is a vector of the reward recieved
    after each transition.
    """
    def add_samples(self, X_t, X_tp1, R, actions):
        raise NotImplementedError()
    
    
    """ 
    Remove a previously add sample. indices provides the index of the sample
    to remove.
    """
    def remove_samples(self, indices):
        raise NotImplementedError()
    
    """
    Get the vector of the rewards for all samples corresponding to a specific action.
    """
    def get_R(self, action):
        raise NotImplementedError()
    
    """
    Set a new reward for all samples corresponding to a given action. 
    Must have an entry for all samples.
    """
    def set_R(self, action):
        raise NotImplementedError()
    
    """
    Returns all the samples stored by the model. If the model is set to discard
    samples after compression, this will throw an exception.
    """
    def get_samples(self):
        raise NotImplementedError()
    
    
    """
    Return a model embedded on the right singular vectors of the sampled
    start states.
    """
    def generate_embedded_model(self, 
                                rank = None, 
                                Kab = None, 
                                Da = None, 
                                wa = None,
                                Vas = None):
        if self.compress_end_states:
            raise NotImplementedError()     

        k = self.num_actions
        for a in xrange(k):
             Kab, Da, wa, Vas = self.update_embedded_model(rank = rank, 
                                                           action = a, 
                                                           Kab = Kab, 
                                                           Da = Da, 
                                                           wa = wa,
                                                           Vas = Vas)
             
        return Kab, Da, wa, Vas
        
    def update_embedded_model(self, action, Kab, Da, wa, Vas, rank = None):
        if self.compress_end_states:
            raise NotImplementedError()     
        if rank is None:
            rank = self.max_rank
        else:
            rank = min(self.max_rank, rank)
        
        lamb = self.lamb
        Ma = self.Ma
        k = self.num_actions
        
        if Da is None:
            Da = np.empty((k,), dtype='O')
        if wa is None:        
            wa = np.empty((k,), dtype='O')
        if Kab is None:
            Kab = np.empty((k,k), dtype='O')
        if Vas is None:
            Vas = np.empty((k,), dtype='O')
        
        a = action
        Ua, Sa, Va = Ma[a]
        Ua = Ua[:,:rank]
        Sa = Sa[:rank]
        Va = Va[:,:rank]
        
        Vas[action] = Va.copy()
        
        Da[a] = Sa/(Sa**2 + lamb)
        wa[a] = self.Ra[a].get_matrix(Ua.shape[0]).squeeze().dot(Ua)
        
        for b in xrange(k):
            Ub, Sb, Vb = Ma[b]
            Ub = Ub[:,:rank]
            Vb = Vb[:,:rank]
            X_tp1 = self.Xa_tp1[b].get_matrix(Ub.shape[0])
            Kab[a,b] = Va.T.dot(X_tp1.T.dot(Ub))
                
        return Kab, Da, wa, Vas
                
            
class BufferedRowCollector:
    def __init__(self, column_dim, buffer_size = 1000, rows = None):
        self.matrix = np.zeros((0, column_dim))
        self.buffer = np.zeros((buffer_size, column_dim))
        self.count = 0
        if rows is not None:
            self.add_rows(rows)
    def add_rows(self, rows):
        if rows.ndim == 1:
            rows = rows.reshape((1,-1))
        while(rows.shape[0]>0):
            j = min(rows.shape[0] + self.count, self.buffer.shape[0])
            self.buffer[self.count:j] = rows[:j-self.count,:]
            rows = rows[j-self.count:, :]
            self.count = j
            if j >= self.buffer.shape[0]:
                self.matrix = np.vstack((self.matrix, self.buffer))
                self.count = 0
                
    def get_matrix(self, num_rows = None):
        if (not num_rows is None) and (num_rows > self.matrix.shape[0] + self.count):
            print self.matrix.shape, self.count, self.buffer.shape
            raise Exception('Asked for too many rows, ' \
            + str(num_rows) + ' rows were asked but only ' \
            + str(self.matrix.shape[0] + self.count) + ' rows are stored.')
        if num_rows is None:
            num_rows = self.matrix.shape[0] + self.count
            
        if num_rows > self.matrix.shape[0]:
            self.matrix = np.vstack((self.matrix, self.buffer[:self.count,:]))
            self.count = 0
            
        return self.matrix[:num_rows,:]
                
                
        
        
