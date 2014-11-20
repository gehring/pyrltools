import numpy as np
import matplotlib.pyplot as plt
import theano

def run_test():
    p = np.arange(-3, 3, 0.2, dtype='float32')
    xx,yy = np.meshgrid(p, p)
    targets = xx**3 + yy**3 + 1
    
    samples = np.hstack((xx.reshape((-1, 1)), yy.reshape((-1,1)))).astype(theano.config.floatX)
    targets = targets.reshape(-1).astype(theano.config.floatX)
    samples = np.vstack((samples, samples))
    targets = np.hstack((targets, targets))
    
    learning_rate = 0.01
    n_hidden = 100
    rng = np.random.RandomState(10)
    l2_coeff= 0.0001
    l1_coeff= 0.0
    i = np.random.randint(samples.shape[0], size=4000)
    
    f_hat = MLPfit(learning_rate, 
                   n_hidden, 
                   rng, 
                   l2_coeff, 
                   l1_coeff, 
                   samples[i], 
                   targets[i],
                   validate_ratio=0.2,
                   n_epochs = 50000,
                   batch_size=120)
    
    y_hat = f_hat(samples)
    print np.mean((y_hat - targets)**2)
    
    p = np.arange(-4, 4, 0.2, dtype='float32')
    xx,yy = np.meshgrid(p, p)
    samples = np.hstack((xx.reshape((-1, 1)), yy.reshape((-1,1)))).astype(theano.config.floatX)
    y_hat = f_hat(samples)
    targets = xx**3 + yy**3 + 1
    
    plt.figure(1)
    plt.subplot(211)
    plt.contourf(p,p, y_hat.reshape((p.size,-1)))
    plt.subplot(212)
    plt.contourf(p,p, targets)
    plt.show()