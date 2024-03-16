from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """


class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """

    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr  # learning rate

    """ Make a step and update all parameters """

    def step(self):
        #### FOR RNN / LSTM ####
        if hasattr(self.net, "preprocess") and self.net.preprocess is not None:
            self.update(self.net.preprocess)
        if hasattr(self.net, "rnn") and self.net.rnn is not None:
            self.update(self.net.rnn)
        if hasattr(self.net, "postprocess") and self.net.postprocess is not None:
            self.update(self.net.postprocess)

        #### MLP ####
        if (
            not hasattr(self.net, "preprocess")
            and not hasattr(self.net, "rnn")
            and not hasattr(self.net, "postprocess")
        ):
            for layer in self.net.layers:
                self.update(layer)


""" Classes """


class SGD(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def update(self, layer):
        for n, dv in layer.grads.items():
            # new_parameter = old_parameter − learning_rate × gradient
            layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def update(self, layer):
        for n, dv in layer.grads.items():
            # velocity = momentum × velocity − learning_rate × gradient
            # new_parameter = old_parameter + velocity
            if n not in self.velocity:
                self.velocity[n] = -self.lr * dv
                
            self.velocity[n] = self.momentum * self.velocity[n] - self.lr * dv

            layer.params[n] += self.velocity[n]


class RMSProp(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def update(self, layer):
        for n, dv in layer.grads.items():
            # init cache
            if n not in self.cache:
                self.cache[n] = np.zeros_like(dv)

            self.cache[n] = self.decay * self.cache[n] + ((1 - self.decay) * (dv ** 2))

            layer.params[n] -= self.lr * dv / np.sqrt(self.cache[n] + self.eps)


class Adam(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def update(self, layer):
        self.t += 1
        for n, dv in layer.grads.items():
            if n not in self.mt:
                self.mt[n] = np.zeros_like(dv)

            if n not in self.vt:
                self.vt[n] = np.zeros_like(dv)

            self.mt[n] = self.beta1 * self.mt[n] + (1 - self.beta1) * dv
            self.vt[n] = self.beta2 * self.vt[n] + (1 - self.beta2) * (dv ** 2)
            mt_prime = self.mt[n]/(1 - (self.beta1 ** self.t))
            vt_prime = self.vt[n] / (1 - (self.beta2 ** self.t))
            layer.params[n] -= self.lr * mt_prime / np.sqrt(vt_prime) + self.eps

