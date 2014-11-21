""" Code taken from theano's tutorial on MLPs """

import numpy
import numpy as np

import time

import theano
import theano.tensor as T
import theano.sparse

from itertools import chain

import sklearn.cross_validation
from sklearn import preprocessing


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.n_out = n_out

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
#         
#         W.tag.test_value = W.get_value()
#         b.tag.test_value = b.get_value()
#         input.tag.test_value = np.random.rand(60, n_in).astype(theano.config.floatX)
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.input = input
        if isinstance(n_hidden, int):
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh
            )
    
            # The logistic regression layer gets as input the hidden units
            # of the hidden layer
            self.outputlayer = HiddenLayer(
                rng=rng,
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out,
                activation=None
            )
             # end-snippet-2 start-snippet-3
            # L1 norm ; one regularization option is to enforce L1 norm to
            # be small
            self.L1 = (
                abs(self.hiddenLayer.W).sum()
                + abs(self.outputlayer.W).sum()
            )
    
            # square of L2 norm ; one regularization option is to enforce
            # square of L2 norm to be small
            self.L2_sqr = (
                (self.hiddenLayer.W ** 2).sum()
                + (self.outputlayer.W ** 2).sum()
            )
    
            # the parameters of the model are the parameters of the two layer it is
            # made out of
            self.params = self.hiddenLayer.params + self.outputlayer.params
        else:
            self.hiddenLayer = [HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden[0],
                activation=T.tanh
            )]
            for n in n_hidden[1:]:
                self.hiddenLayer.append(HiddenLayer(
                rng=rng,
                input=self.hiddenLayer[-1].output,
                n_in=self.hiddenLayer[-1].n_out,
                n_out=n,
                activation=T.tanh
            ))
                # The logistic regression layer gets as input the hidden units
            # of the hidden layer
            self.outputlayer = HiddenLayer(
                rng=rng,
                input=self.hiddenLayer[-1].output,
                n_in=n_hidden[-1],
                n_out=n_out,
                activation=None
            )
            # L1 norm ; one regularization option is to enforce L1 norm to
            # be small
            self.L1 = (
                sum([ abs(h.W).sum() for h in self.hiddenLayer])
                + abs(self.outputlayer.W).sum()
            )
    
            # square of L2 norm ; one regularization option is to enforce
            # square of L2 norm to be small
            self.L2_sqr = (
                sum([ (h.W ** 2).sum() for h in self.hiddenLayer])
                + (self.outputlayer.W ** 2).sum()
            )
    
            # the parameters of the model are the parameters of the two layer it is
            # made out of
            self.params = list(chain(*[h.params for h in self.hiddenLayer])) + self.outputlayer.params
                
       
        
        self.output = self.outputlayer.output
        
    def L2cost(self, y):
        err = self.output[:,0] - y
        return T.mean((err*err))
    
    def getf(self, x):
        return theano.function([x], self.output)

def MLPregression(learning_rate, 
                    mlp, 
                    l2_coeff, 
                    l1_coeff,
                    samples,
                    target,
                    batch_size,
                    validate_ratio,
                    rng,
                    n_epochs= 100,
                    validation_freq = 2,
                    patience = 10000,
                    patience_increase = 2,
                    improvement_thresh = 0.995):
             
    
    
    train_samples, test_samples, train_target, test_target = \
        sklearn.cross_validation.train_test_split(samples, 
                                              target, 
                                              test_size = validate_ratio,
                                              random_state = rng
                                              )
    train_samples = theano.shared(train_samples.astype(theano.config.floatX),borrow = False)
    test_samples = theano.shared(test_samples.astype(theano.config.floatX),borrow = False)
    train_target = theano.shared(train_target.astype(theano.config.floatX),borrow = False)
    test_target = theano.shared(test_target.astype(theano.config.floatX),borrow = False)
    
    
    n_train_batches = train_samples.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_samples.get_value(borrow=True).shape[0] / batch_size
    
    x = mlp.input
    y = T.fvector('y')
#     y.tag.test_value = numpy.random.rand(batch_size*2).astype(theano.config.floatX)
    index = T.lscalar()
    index.tag.test_value = 0
    
    cost = (mlp.L2cost(y) + mlp.L1*l1_coeff + mlp.L2_sqr*l2_coeff)
    
    test_model = theano.function(
            inputs=[index],
            outputs=mlp.L2cost(y),
            givens={
                x: test_samples[index * batch_size:(index + 1) * batch_size],
                y: test_target[index * batch_size:(index + 1) * batch_size]
            }
        )

    gparams = [T.grad(cost, param) for param in mlp.params]
    
    updates = [ (param, param - learning_rate * gparam)
                    for (param, gparam) in zip(mlp.params, gparams)]
    
    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: train_samples[index * batch_size:(index + 1) * batch_size],
            y: train_target[index * batch_size:(index + 1) * batch_size]
            }
        )
    
    epoch = 0
    count = 0

    best_val = numpy.inf
    
    best_params = [ p.get_value(borrow = False) for p in mlp.params]
    done_train = False
    
#     start_time = time.clock()
    while epoch < n_epochs and (not done_train):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_cost = train_model(minibatch_index)
            
            if count %validation_freq == 0:
                val_cost = np.mean([test_model(i) for i in xrange(n_test_batches)])
                
                if val_cost < best_val:
                    if val_cost < best_val*improvement_thresh:
                        patience = max(patience, count*patience_increase)
                    best_val = val_cost
#                     print best_val
                    best_params = [ p.get_value(borrow = False) for p in mlp.params]
                    
                    
            # number of minibatches seen
            count += 1
            
            # termination condition
            done_train = patience <= count
    
#     end_time = time.clock()
#     print 'Time taken for opt: ' + str(end_time - start_time) + 's'
    for p, best_p in zip(mlp.params, best_params):
        p.set_value(best_p)


    train_samples.set_value([[]])
    train_target.set_value([])
    test_samples.set_value([[]])
    test_target.set_value([])
    return mlp.getf(x)

def MLPSparseRegression(learning_rate, 
                    mlp, 
                    l2_coeff, 
                    l1_coeff,
                    samples,
                    target,
                    batch_size,
                    validate_ratio,
                    rng,
                    n_epochs= 100,
                    validation_freq = 2,
                    patience = 10000,
                    patience_increase = 2,
                    improvement_thresh = 0.995):
             
    scaler = preprocessing.StandardScaler.fit(samples)
    samples = scaler.transform(samples)
    
    train_samples, test_samples, train_target, test_target = \
        sklearn.cross_validation.train_test_split(samples, 
                                              target, 
                                              test_size = validate_ratio,
                                              random_state = rng
                                              )
    train_samples = theano.sparse.shared(train_samples.astype(theano.config.floatX),borrow = False)
    test_samples = theano.sparse.shared(test_samples.astype(theano.config.floatX),borrow = False)
    train_target = theano.shared(train_target.astype(theano.config.floatX),borrow = False)
    test_target = theano.shared(test_target.astype(theano.config.floatX),borrow = False)
    
    
    n_train_batches = train_samples.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_samples.get_value(borrow=True).shape[0] / batch_size
    
    x = mlp.input
    y = T.fvector('y')
#     y.tag.test_value = numpy.random.rand(batch_size*2).astype(theano.config.floatX)
    index = T.lscalar()
    index.tag.test_value = 0
    
    cost = (mlp.L2cost(y) + mlp.L1*l1_coeff + mlp.L2_sqr*l2_coeff)
    
    test_model = theano.function(
            inputs=[index],
            outputs=mlp.L2cost(y),
            givens={
                x: test_samples[index * batch_size:(index + 1) * batch_size],
                y: test_target[index * batch_size:(index + 1) * batch_size]
            }
        )

    gparams = [T.grad(cost, param) for param in mlp.params]
    
    updates = [ (param, param - learning_rate * gparam)
                    for (param, gparam) in zip(mlp.params, gparams)]
    
    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: train_samples[index * batch_size:(index + 1) * batch_size],
            y: train_target[index * batch_size:(index + 1) * batch_size]
            }
        )
    
    epoch = 0
    count = 0

    best_val = numpy.inf
    
    best_params = [ p.get_value(borrow = False) for p in mlp.params]
    done_train = False
    
#     start_time = time.clock()
    while epoch < n_epochs and (not done_train):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_cost = train_model(minibatch_index)
            
            if count %validation_freq == 0:
                val_cost = np.mean([test_model(i) for i in xrange(n_test_batches)])
                
                if val_cost < best_val:
                    if val_cost < best_val*improvement_thresh:
                        patience = max(patience, count*patience_increase)
                    best_val = val_cost
#                     print best_val
                    best_params = [ p.get_value(borrow = False) for p in mlp.params]
                    
                    
            # number of minibatches seen
            count += 1
            
            # termination condition
            done_train = patience <= count
    
#     end_time = time.clock()
#     print 'Time taken for opt: ' + str(end_time - start_time) + 's'
    for p, best_p in zip(mlp.params, best_params):
        p.set_value(best_p)


    train_samples.set_value([[]])
    train_target.set_value([])
    test_samples.set_value([[]])
    test_target.set_value([])
    return lambda x: mlp.getf(scaler.transform(x))


def MLPfit(learning_rate, 
            n_hidden,
            rng, 
            l2_coeff, 
            l1_coeff,
            samples,
            target,
            batch_size = 100,
            validate_ratio=0.1,
            n_epochs= 1000,
            validation_freq = 2,
            patience = 10000,
            patience_increase = 2,
            improvement_thresh = 0.995):
    x = T.matrix('x')
#     x.tag.test_value =  numpy.random.rand(batch_size*2,samples.shape[1]).astype(theano.config.floatX)
    mlp = MLP(rng, 
              x,
              samples.shape[1], 
              n_hidden, 
              1)
    return MLPregression(learning_rate, 
                         mlp, 
                         l2_coeff, 
                         l1_coeff, 
                         samples, 
                         target, 
                         batch_size, 
                         validate_ratio, 
                         rng, 
                         n_epochs, 
                         validation_freq, 
                         patience, 
                         patience_increase, 
                         improvement_thresh)
    
def MLPSparsefit(learning_rate, 
            n_hidden,
            rng, 
            l2_coeff, 
            l1_coeff,
            samples,
            target,
            batch_size = 100,
            validate_ratio=0.1,
            n_epochs= 1000,
            validation_freq = 2,
            patience = 10000,
            patience_increase = 2,
            improvement_thresh = 0.995):
    x = theano.sparse.matrix('x')
#     x.tag.test_value =  numpy.random.rand(batch_size*2,samples.shape[1]).astype(theano.config.floatX)
    mlp = MLP(rng, 
              x,
              samples.shape[1], 
              n_hidden, 
              1)
    return MLPregression(learning_rate, 
                         mlp, 
                         l2_coeff, 
                         l1_coeff, 
                         samples, 
                         target, 
                         batch_size, 
                         validate_ratio, 
                         rng, 
                         n_epochs, 
                         validation_freq, 
                         patience, 
                         patience_increase, 
                         improvement_thresh)



                
    