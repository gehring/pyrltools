import numpy
import numpy as np
from itertools import product
from numpy import random

class Logisticfn(object):
    def __init__(self):
        pass

    def evaluate(self, x):
        return 1.0/( 1 + np.exp(-x))

    def evaluatederiv(self, x, out=None):
        fx = self.evaluate(x)
        return fx * (1-fx)

    def evaluatederivderiv(self, x, out=None):
        emx = np.exp(x)
        return emx * (emx-1)/ (emx+1)**3

class Linearfn(object):
    def __init__(self):
        pass

    def evaluate(self, x, out=None):
        if out == None:
            out = numpy.zeros_like(x)
        out = x
        return out

    def evaluatederiv(self, x, out=None):
        if out == None:
            out = numpy.ones_like(x)
        else:
            out[:] = 1
        return out

    def evaluatederivderiv(self, x, out=None):
        if out == None:
            out = numpy.zeros_like(x)
        else:
            out[:] = 0
        return  out

class LinearRectifier(object):
    def __init__(self):
        pass

    def evaluate(self, x, out=None):
        if out == None:
            out = numpy.zeros_like(x)
        out[:] = x
        numpy.clip(out, 0, float('inf'), out= out)
        return out

    def evaluatederiv(self, x, out=None):
        if out == None:
            out = numpy.zeros_like(x)

        out[:] = x
        out[:] *= float('inf')
        numpy.clip(out, 0, 1, out= out)
        return out

    def evaluatederivderiv(self, x, out=None):
        if out == None:
            out = numpy.zeros_like(x)
        else:
            out[:] = 0
        return  out


class NeuronLayer(object):
    def __init__(self, input_size, layer_input, num_neuron, sigmoid, **argk):
        self.a = numpy.zeros(num_neuron)

        init_bias_var = argk.get('init_bias_var', 0.1)
        init_w_var = argk.get('init_w_var', 0.01)

        self.w = numpy.random.normal(0, init_w_var, (num_neuron, layer_input))
        self.bias = numpy.random.normal(0, init_bias_var, num_neuron)

        self.psi = numpy.zeros((num_neuron, input_size))
        self.gradout = numpy.zeros((num_neuron, input_size))

        self.out = numpy.zeros(num_neuron)
        self.sigmoid = sigmoid

        self.mommentum = argk.get('mommentum', 0.0)
        self.prev_dw = None
        self.prev_dbias = None

    def compute_gradient(self, errors_sig, errors_gradsig):
        dsigmoid = self.sigmoid.evaluatederiv(self.a)
        ddsigmoid = self.sigmoid.evaluatederivderiv(self.a)

        # first part is the vanilla backprog, second part is to account for the
        # errors induced by the gradient
        deda = ddsigmoid * numpy.sum(self.psi*errors_gradsig, axis=0)
        deda += dsigmoid * errors_sig

        dedpsi = errors_gradsig * dsigmoid

        # build the error gradient
        dedw = dedpsi.T.dot(self.input_grad)

        tmp1 = numpy.empty_like(dedw)
        tmp1[:] = self.input
        tmp2 = tmp1.T
        tmp2 *= deda

        dedw += tmp1

        dbias = deda

        # propagate errors to inputs
        dedinput = self.w.T.dot(deda)
        dedgradin = dedpsi.dot(self.w)

        return dedw, dbias, dedinput, dedgradin

    def update_weights(self, dedw, dbias):
        if self.prev_dw == None:
            self.prev_dw = -dedw
            self.prev_dbias = -dbias
        else:
            self.prev_dw = self.prev_dw * self.mommentum - dedw
            self.prev_dbias = self.prev_dbias * self.mommentum - dbias
        self.w += self.prev_dw
        self.bias += self.prev_dbias

    def evaluate(self, inputs, grad):
        self.input = inputs
        self.input_grad = grad
        self.a = self.w.dot(inputs) + self.bias
        self.out = self.sigmoid.evaluate(self.a)

        dsigmoid = self.sigmoid.evaluatederiv(self.a)
        self.psi = grad.dot(self.w.T)
        self.gradout = self.psi * dsigmoid

        return self.out, self.gradout


class NeuralNet(object):
    def __init__(self, layers, **kargs):
        self.alpha = kargs.get('alpha', 0.05)
        self.eta = kargs.get('eta', 0.0)
        self.alpharatio = kargs.get('alpharatio', 1.0)

        sigmoid = Logisticfn() #LinearRectifier()
        self.layers = [ NeuronLayer(layers[0], layers[i], layers[i+1], sigmoid, **kargs)
                            for i in range(len(layers) - 1)]
        self.layers[-1].sigmoid = Linearfn()

    def evaluate(self, inputs):
        grad = numpy.eye(len(inputs))
        for l in self.layers:
            inputs, grad = l.evaluate(inputs, grad)
        return self.layers[-1].out

    def backprop(self, target, direction, dirderiv):
        norm = numpy.linalg.norm(direction)
        if norm > 0:
            direction = direction/norm
            dirderiv /= norm

        dedinput = (1-self.eta) * (self.layers[-1].out - target)
        dedgradin = numpy.array([(self.eta* (direction.dot(self.layers[-1].gradout) - dirderiv)
                        * direction)]).T

        err_grad = []
        for l in reversed(self.layers):
            dedw, dedb, dedinput, dedgradin = l.compute_gradient(dedinput, dedgradin)
            err_grad.append((dedw, dedb))
        alpha = self.alpha
        for l, grad in zip(reversed(self.layers), err_grad):
            l.update_weights(grad[0] * self.alpha, grad[1] * alpha)
            alpha *= self.alpharatio

    def getgradient(self, target, direction, dirderiv):
#         norm = numpy.linalg.norm(direction)
#         if norm > 0:
#             direction = direction/norm
#             dirderiv /= norm
        dedinput = (1-self.eta) * (self.layers[-1].out - target)
        dedgradin = numpy.array([(self.eta* (direction.dot(self.layers[-1].gradout) - dirderiv)
                        * direction)]).T
        err_grad = []
        for l in reversed(self.layers):
            dedw, dedb, dedinput, dedgradin = l.compute_gradient(dedinput, dedgradin)
            err_grad.append((dedw, dedb))
        return reversed(err_grad)










