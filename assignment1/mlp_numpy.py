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
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Linear module - maps from n_inputs(=10) to n_hidden(=128)
        self.lin_mod1 = LinearModule(in_features=n_inputs, out_features=n_hidden[0], input_layer=True) # TODO: not sure what kind of difference input layer makes, likely only has to do with initialization of weights?

        # ELU
        self.elu = ELUModule(alpha=1) # TODO: Make choice on (1) hard-code alpha (2 - favorite ) be able to choose it in init as hyperparam (3) other?
        # Note: pytorch uses standard value of alpha=1, so we use that here as well.

        # Linear module - maps from n_hidden(=128) to n_classes(=5)
        self.lin_mod2 = LinearModule(in_features=n_hidden[0], out_features=n_classes)

        # Softmax activation
        self.softmax = SoftMaxModule()

        #######################
        # END OF YOUR CODE    #
        #######################

    def __call__(self, args, *kwargs):
        return self.forward(args, *kwargs) # custom call to make our class more similar to pytorch

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        # Linear module - maps from n_inputs(=10) to n_hidden(=128)
        z1 = self.lin_mod1.forward(x)

        # ELU
        a1_elu = self.elu.forward(z1)

        # Linear module - maps from n_hidden(=128) to n_classes(=5)
        z2 = self.lin_mod2.forward(a1_elu)

        # Softmax activation
        out = self.softmax.forward(z2)       

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss
        """

        #######################
        # PUT YOUR CODE HERE  #
        ####################### 
                
        # calculate dz2
        dz2 = self.softmax.backward(dout=dout) # updates W2

        # calculate d a1_elu
        da1_elu = self.lin_mod2.backward(dout=dz2)

        # calculate dz1
        dz1 = self.elu.backward(dout=da1_elu)

        # we COULD calculate dx here, but don't need to because all weights and biases have been updated
        _ = self.lin_mod1.backward(dout=dz1)

        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.lin_mod1.clear_cache()
        self.lin_mod2.clear_cache()
        self.elu.clear_cache()
        self.softmax.clear_cache
        #######################
        # END OF YOUR CODE    #
        #######################
