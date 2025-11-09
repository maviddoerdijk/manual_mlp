################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients


        # initialize a weight matrix of zeros that maps from (X x in_features) to (X x out_features)
        kaiming_mean = 0
        magic_number = 1.0 if input_layer else 2.0
        kaiming_std_dev = np.sqrt(magic_number/in_features)  # number of input connections to that neuron
        W = np.random.normal(loc=kaiming_mean, scale=kaiming_std_dev, size=(out_features, in_features))
        self.params['weight'] = W

        # initialize a bias matrix of zeros with the same shape as output vector
        B = np.zeros(shape=(out_features, ))
        self.params['bias'] = B

        ## gradients
        self.grads['weight'] = np.zeros(shape=(out_features, in_features))
        self.grads['bias'] = np.zeros(shape=(out_features, ))

        self.cache = None
        


    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """



        # z = xW^T + b
        out = x @ self.params['weight'].T + self.params['bias']  # shape (batch_size, out_features)
        
        self.cache = x # store, because we will later do backward pass using d../d.. = ...



        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """


        X = self.cache
        W = self.params['weight']

        ## Update parameters
        # param 1: dL/dW = dZ/dW . dL/dZ = A^T . dout
        self.grads['weight'] = dout.T @ X
        # Note: I will assume we are not expected to accumulate gradients, even though it is default pytorch behavior. Because we are not given a function for resetting gradients to zero.  

        # param 2: dL/dB = dL/dZ . dZ/dB = (1 1 ... 1)^T . dout
        self.grads['bias'] = np.sum(dout, axis=0)  # sum over batch dimension


        ## Compute dx, the gradient that will be passed further to the previous layer
        # dL/dZ = dL/dout . dout/dZ = dout . W
        dx = dout @ W


        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        self.cache = None

class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        

        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1)) # following formula found in https://www.geeksforgeeks.org/deep-learning/elu-activation-function-in-neural-network/

        self.cache = x



        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """


        Z = self.cache

        local_grad = np.where(Z > 0, 1, self.alpha * np.exp(Z))

        ## Compute dx, the gradient that will be passed further to the previous layer
        # dx = (in more complete syntax) dL/dx = dL/dout . dout/dx = ...
        dx = np.multiply(dout, local_grad)


        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """

        self.cache = None



class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """


        def shift_and_exp(x):
          shift_value = np.max(x) # shift by scalar b such that the softmax output becomes shift-invariant
          y = np.exp(x - shift_value) # reasonable choice according to https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
          return y

        # 1. get matrix of row sums of same shape as x
        x_shifted_and_exp = shift_and_exp(x) # e to the power each entry
        x_shifted_and_exp_row_sum = np.sum(x_shifted_and_exp, axis=1, keepdims=True) # sum over exponents, should collapse from (S x N) to (S, )
        x_shifted_and_exp_row_sum_matrix = np.hstack((x_shifted_and_exp_row_sum, ) * x.shape[1]) # populate with N times the second dim (many times the same sum of exponents)

        # 2. do pair-wise division of x by matrix_row_sum
        if x_shifted_and_exp.shape != x_shifted_and_exp_row_sum_matrix.shape:
            raise ValueError("Mistake in implementation: Shape of input does not match shape of element-wise division for softmax.")
        out = x_shifted_and_exp / x_shifted_and_exp_row_sum_matrix
        
        self.cache = out


        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
        """


        ## Compute dx, the gradient that will be passed further to the previous layer
        # dx = (in more complete syntax) dL/dx = dL/dout . dout/dx = (Y_pred - Y_onehot) / batch_size
        s = self.cache
        dot_product = np.sum(dout * s, axis=1, keepdims=True)
        dx = s * (dout - dot_product)



        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """

        self.cache = None



class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
        """


        # turn y into a one-hot encoded matrix with same shape as x
        batch_size = x.shape[0]
        n_classes = x.shape[1]
        y_onehot = np.eye(n_classes)[y]
        
        epsilon = 1e-8 # to avoid log(0)
        ce_entries = y_onehot * np.log(x + epsilon)

        # get loss for each sample in the batch
        out = - np.sum(ce_entries) / batch_size 



        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
        """
        # get values from cache
        Y_pred = x
        n_classes = x.shape[1]
        Y_onehot = np.eye(n_classes)[y] 
        batch_size = x.shape[0]


        ## Compute dx, the gradient that will be passed further to the previous layer
        # dx = (in more complete syntax) dL/dx = dL/dout . dout/dx = (Y_pred - Y_onehot) / batch_size
        dx = -(Y_onehot / Y_pred) / batch_size



        return dx