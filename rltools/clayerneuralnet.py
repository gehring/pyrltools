import numpy as np
from ctypes import *
import rltools.ext_neuro as ext_neuro
import sys

class CMatrix(Structure):
    _fields_ = [('data', POINTER(c_double)),
                ('m', c_uint),
                ('n', c_uint),
                ('size', c_uint)]

class CNLayer(Structure):
    _fields_ = [ ('w', POINTER(CMatrix)),
                ('bias', POINTER(CMatrix)),
                ('a', POINTER(CMatrix)),
                ('psi', POINTER(CMatrix)),
                ('out', POINTER(CMatrix)),
                ('out_grad', POINTER(CMatrix)),
                ('input', POINTER(CMatrix)),
                ('in_grad', POINTER(CMatrix)),
                ('deda', POINTER(CMatrix)),
                ('dedw', POINTER(CMatrix)),
                ('prev_dw', POINTER(CMatrix)),
                ('dedpsi', POINTER(CMatrix)),
                ('dbias', POINTER(CMatrix)),
                ('prev_dbias', POINTER(CMatrix)),
                ('dedinput', POINTER(CMatrix)),
                ('dedgradin', POINTER(CMatrix)),
                ('mommentum', c_double),
                ('sig_eval', CFUNCTYPE(POINTER(c_double), POINTER(c_double))),
                ('sig_deval', CFUNCTYPE(POINTER(c_double), POINTER(c_double))),
                ('sig_ddeval', CFUNCTYPE(POINTER(c_double), POINTER(c_double)))]

# ext_neuro = CDLL('C lib/neurosftd_lib.so')
# # ext_neuro.compute_gradient_from_np.argtypes = (POINTER(CNLayer), POINTER(object), POINTER(object))
# # ext_neuro.create_layer.argtypes = (c_uint, c_uint, c_uint, POINTER(c_double),
# #                                      POINTER(c_double), c_double,
# #                                      CFUNCTYPE(POINTER(c_double), POINTER(c_double)),
# #                                      CFUNCTYPE(POINTER(c_double), POINTER(c_double)),
# #                                      CFUNCTYPE(POINTER(c_double), POINTER(c_double)))
# # ext_neuro.create_matrix.argtypes = (c_uint, c_uint)
# # ext_neuro.destroy_matrix.argtypes = (POINTER(CMatrix))
# # ext_neuro.destroy_layer.argtypes = (POINTER(CNLayer))
#
callback_type = CFUNCTYPE(c_double, c_double)


class Logisticfn(object):
    def __init__(self):
        pass

    def evaluate(self, x, out=None):
        if out == None:
            out = np.zeros_like(x)
        np.exp(-x, out=out)
        out += 1
        out **= -1
        return out

    def evaluatederiv(self, x, out=None):
        if out == None:
            out = np.zeros_like(x)

        self.evaluate(x, out=out)
        out -= out**2
        return out

    def evaluatederivderiv(self, x, out=None):
        if out == None:
            out = np.zeros_like(x)

        ex = np.exp(x)
        out[:] = (- ex*( ex - 1)/(ex+1)**3)
#         np.exp(x, out=out)
#         ex = np.array(out)
#         out *= - (ex -1)
#         out /= (ex + 1)**3
        return  out

class Linearfn(object):
    def __init__(self):
        pass

    def evaluate(self, x, out=None):
        if out == None:
            out = np.zeros_like(x)
        out += x
        return out

    def evaluatederiv(self, x, out=None):
        if out == None:
            out = np.ones_like(x)
        else:
            out[:] = 1
        return out

    def evaluatederivderiv(self, x, out=None):
        if out == None:
            out = np.zeros_like(x)
        else:
            out[:] = 0
        return  out

class LinearRectifier(object):
    def __init__(self):
        pass

    def evaluate(self, x, out=None):
        if out == None:
            out = np.zeros_like(x)
        out[:] = x
        np.clip(out, 0, float('inf'), out= out)
        return out

    def evaluatederiv(self, x, out=None):
        if out == None:
            out = np.zeros_like(x)

        out[:] = x
        out[:] *= float('inf')
        np.clip(out, 0, 1, out= out)
        return out

    def evaluatederivderiv(self, x, out=None):
        if out == None:
            out = np.zeros_like(x)
        else:
            out[:] = 0
        return  out


def mse(arr1, arr2):
    d = arr1 - arr2
    return np.sqrt(np.mean(d**2))

class RBFLayer(object):
    def __init__(self,
                 input_size,
                 layer_input,
                 num_neuron,
                 in_range = np.array([0,1])):

        if in_range.ndim>1:
            self.c = np.array([[ np.random.uniform(in_range[i][0], in_range[i][1])
                                        for i in range(layer_input)]
                                        for j in range(num_neuron)])
        else:
            self.c = np.array([np.random.uniform(in_range[0], in_range[1],
                                                 size = layer_input)
                                for j in range(num_neuron)])

    def evaluate(self, inputs, grad):
        diff = self.c - inputs
        a = -np.sum(diff**2, axis=0)
        self.out = np.exp(a)
        tmp = 2 * self.out * diff
        self.out_grad = tmp.dot(grad)
        return self.out, self.out_grad


    def compute_gradient(self, errors_sig, errors_gradsig):
        return None, None, None, None

    def update_weights(self, dedw, dbias):
        pass


class NeuronLayer(object):
    def __init__(self,
                 input_size,
                 layer_input,
                 num_neuron,
                 sigmoid,
                 sig,
                 sigd,
                 sigdd,
                 **argk):

        mommentum = argk.get('mommentum', 0.9)
        size_w = num_neuron * layer_input
#         init_w = np.random.uniform(-0.1, 0.1, size_w)
        init_w = np.random.normal(0, 0.01, size_w)
        size_b = num_neuron * layer_input
#         init_b = np.random.uniform(-0.5, 0.5, size_b)
        init_b = np.random.normal(0, 1, size_b)


        self.sigmoid = sigmoid
        self.mommentum = mommentum

        self.cnlayer = ext_neuro.create_layer(input_size,
                                                layer_input,
                                                num_neuron,
                                                init_w,
                                                init_b,
                                                mommentum,
                                                sig,
                                                sigd,
                                                sigdd)
        self.a = ext_neuro.get_a(self.cnlayer)[:,0]
        self.deda = ext_neuro.get_deda(self.cnlayer)[:,0]

        self.w = ext_neuro.get_w(self.cnlayer)
        self.bias = ext_neuro.get_bias(self.cnlayer)[:,0]

        self.dedinput = ext_neuro.get_dedinput(self.cnlayer)[:,0]
        self.dedgradin = ext_neuro.get_dedgradin(self.cnlayer)

        self.psi = ext_neuro.get_psi(self.cnlayer)
        self.dedpsi = ext_neuro.get_dedpsi(self.cnlayer)
        self.gradout = ext_neuro.get_out_grad(self.cnlayer)

        self.out = ext_neuro.get_out(self.cnlayer)[:,0]
        self.input = ext_neuro.get_input(self.cnlayer)[:,0]
        self.input_grad = ext_neuro.get_in_grad(self.cnlayer)

        self.prev_dw = ext_neuro.get_prev_dw(self.cnlayer)
        self.prev_dbias = ext_neuro.get_prev_dbias(self.cnlayer)[:,0]
        self.dedw = ext_neuro.get_dedw(self.cnlayer)
        self.dbias = ext_neuro.get_dbias(self.cnlayer)[:,0]

    def compute_gradient(self, errors_sig, errors_gradsig):
        ext_neuro.compute_gradient_from_np(self.cnlayer,
                                             errors_sig,
                                             errors_gradsig)

        dsigmoid = self.sigmoid.evaluatederiv(self.a)
        ddsigmoid = self.sigmoid.evaluatederivderiv(self.a)

        # first part is the vanilla backprog, second part is to account for the
        # errors induced by the gradient
        deda = ddsigmoid * np.sum(self.psi*errors_gradsig, axis=0)
        deda += dsigmoid * errors_sig

        dedpsi = errors_gradsig * dsigmoid

        # build the error gradient
        dedw = dedpsi.T.dot(self.input_grad)

        tmp1 = np.empty_like(dedw)
        tmp1[:] = self.input
        tmp2 = tmp1.T
        tmp2 *= deda

        dedw += tmp1

        dbias = deda

        # propagate errors to inputs
        dedinput = self.w.T.dot(deda)
        dedgradin = dedpsi.dot(self.w)
#
        thresh = 1.0E-6
#         print deda
#         print self.deda
        assert np.linalg.norm(self.deda - deda) < thresh
#         print dedpsi
#         print self.dedpsi
        if np.linalg.norm(self.dedpsi - dedpsi) > thresh:
            print self.dedpsi
            print dedpsi
            assert np.linalg.norm(self.dedpsi - dedpsi) < thresh
#         print dedw
#         print self.dedw
        assert np.linalg.norm(self.dedw - dedw) < thresh
#         print dbias
#         print self.dbias
        assert np.linalg.norm(self.dbias - dbias) < thresh
#         print dedinput
#         print self.dedinput
        assert np.linalg.norm(self.dedinput - dedinput) < thresh
#         print dedgradin
#         print self.dedgradin
        assert np.linalg.norm(self.dedgradin - dedgradin) < thresh


        return self.dedw, self.dbias, self.dedinput, self.dedgradin


    def update_weights(self, dedw, dbias):
        self.prev_dw[:] = self.prev_dw * self.mommentum + dedw
        self.prev_dbias[:] = self.prev_dbias * self.mommentum + dbias
        self.w -= self.prev_dw
        self.bias -= self.prev_dbias

    def evaluate(self, inputs, grad):
        ext_neuro.evaluate_layer_from_np(self.cnlayer,
                                             inputs,
                                             grad)
#
#         assert np.linalg.norm(inputs - self.input) < 1.0e-8
#         assert np.linalg.norm(grad - self.input_grad) < 1.0e-8

        return self.out, self.gradout

    def __del__(self):
        ext_neuro.destroy_layer(self.cnlayer)

class NeuralNet(object):
    def __init__(self, layers, **kargs):
        self.alpha = kargs.get('alpha', 0.01)
        self.eta = kargs.get('eta', 0.0)

        sigmoid = Logisticfn()

        self.layers = [ NeuronLayer(layers[0],
                                    layers[i],
                                    layers[i+1],
                                    sigmoid,
                                    ext_neuro.get_logistic_sig(0),
                                    ext_neuro.get_logistic_sig(1),
                                    ext_neuro.get_logistic_sig(2),
                                    **kargs)
                            for i in range(len(layers) - 2)]

        self.layers.append(NeuronLayer(layers[0],
                                    layers[-2],
                                    layers[-1],
                                    Linearfn(),
                                    ext_neuro.get_linear_sig(0),
                                    ext_neuro.get_linear_sig(1),
                                    ext_neuro.get_linear_sig(2),
                                    **kargs))

    def evaluate(self, inputs):
        grad = np.eye(len(inputs))
        inputs = np.array(inputs)
        for l in self.layers:
#             ext_neuro.evaluate_layer_from_np(l.cnlayer, inputs, grad)
#             inputs = l.out
#             grad = l.gradout
            inputs, grad = l.evaluate(inputs, grad)
        return self.layers[-1].out

    def backprop(self, target, direction, dirderiv):
        norm = np.linalg.norm(direction)
        direction = direction/norm
        dirderiv /= norm

        dedinput = (1-self.eta) * (self.layers[-1].out - target)
        dedgradin = np.array([(self.eta* (direction.dot(self.layers[-1].gradout) - dirderiv)
                        * direction)]).T

        err_grad = []
        for l in reversed(self.layers):
#             ext_neuro.compute_gradient_from_np(l.cnlayer,
#                                              dedinput,
#                                              dedgradin)
#             ext_neuro.update_weights_from_py(l.cnlayer, self.alpha)
#             dedw = l.dedw
#             dedb = l.dbias
#             dedinput = l.dedinput
#             dedgradin = l.dedgradin
            dedw, dedb, dedinput, dedgradin = l.compute_gradient(dedinput, dedgradin)
            err_grad.append((dedw, dedb))

        for l, grad in zip(reversed(self.layers), err_grad):
            if( grad[0] != None):
                l.update_weights(grad[0] * self.alpha, grad[1] * self.alpha)

    def getgradient(self, target, direction, dirderiv):
        norm = np.linalg.norm(direction)
        direction = direction/norm
        dirderiv /= norm

        dedinput = (1-self.eta) * (self.layers[-1].out - target)
        dedgradin = np.array([(self.eta* (direction.dot(self.layers[-1].gradout) - dirderiv)
                        * direction)]).T


        err_grad = []
#         for l in reversed(self.layers):
#             ext_neuro.compute_gradient_from_np(l.cnlayer,
#                                              dedinput,
#                                              dedgradin)
#             dedinput = l.dedinput
#             dedgradin = l.dedgradin
#             err_grad.append( (l.dedw, l.dbias))

        for l in reversed(self.layers):
        #             ext_neuro.compute_gradient_from_np(l.cnlayer,
        #                                              dedinput,
        #                                              dedgradin)
        #             ext_neuro.update_weights_from_py(l.cnlayer, self.alpha)
        #             dedw = l.dedw
        #             dedb = l.dbias
        #             dedinput = l.dedinput
        #             dedgradin = l.dedgradin
            dedw, dedb, dedinput, dedgradin = l.compute_gradient(dedinput, dedgradin)
            err_grad.append((dedw, dedb))
        return reversed(err_grad)










