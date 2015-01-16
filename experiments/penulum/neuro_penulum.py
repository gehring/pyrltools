from rltools.theanotools import NeuroSFTD, sym_RBF
from rltools.SwingPendulum import SwingPendulum

import theano
import theano.tensor as T

import numpy as np

domain = SwingPendulum(random_start=True)
s_range = domain.state_range

ds = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,False))('ds')
s = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,False))('s')



xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10),
                     np.linspace(s_range[0][0], s_range[1][0], 10))

centers = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)

widths = (s_range[1]-s_range[0]).astype(theano.config.floatX)*0.2

x,_,size = sym_RBF(s, s, ds, centers, widths, bias_term=True)
neurosftd = NeuroSFTD(x, 
                      size, 
                      [40], 
                      np.random.RandomState(1), 
                      alpha=0.05, 
                      alpha_mu=0.05, 
                      eta=0.0, 
                      beta_1=0.01, 
                      beta_2=0.05)

