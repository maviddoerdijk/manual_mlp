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
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
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
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.
        """
        super(MLP, self).__init__() # inherit everything from nn.Module


        features = []
        if not n_hidden:
            features.append(nn.Linear(in_features=n_inputs, out_features=n_classes)) # without hidden layers, just map from n_inputs to n_classes
        else:
            features.append(nn.Linear(in_features=n_inputs, out_features=n_hidden[0]))
            if use_batch_norm:
                features.append(nn.BatchNorm1d(num_features=n_hidden[0])) # add batchnorm after fc layer, before activation layer
            features.append(nn.ELU(alpha=1.0))
            for i in range(len(n_hidden) - 1): # map from i to i-1 (last iteration maps from n_hidden[-2] to n_hidden[-1])
                features.append(nn.Linear(in_features=n_hidden[i], out_features=n_hidden[i+1]))
                if use_batch_norm:
                    features.append(nn.BatchNorm1d(num_features=n_hidden[i+1])) # add batchnorm after fc layer, before activation layer
                features.append(nn.ELU(alpha=1.0))
            features.append(nn.Linear(in_features=n_hidden[-1], out_features=n_classes)) # # add one last mapping from n_hidden[-1] to n_classes. Note: This outputs logits for each class, rather than softmax probabilities. This is fine as the nn.CrossEntropyLoss expects logits. 
        self.features = nn.Sequential(*features)

        # initialize weights and biases as requested
        j = 0
        for layer in features:
            if isinstance(layer, nn.Linear):
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias) # also according to ed
                if j == 0:
                  # logic uses std = gain / sqrt(fan_mode). a = 1.0 -> gain = sqrt(2/(1+a^2)) = 1 -> magic_number = gain^2 = 1   
                  nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu', a=1.0)
                else:
                  # relu -> a=0 -> gain = sqrt( 2 / (1+a^2) ) = sqrt(2) -> gain^2 = magic_number = 2
                  nn.init.kaiming_normal_(layer.weight, nonlinearity='relu') # Ed discussion mentions we should do it this way
            j += 1




    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """


        out = self.features(x) # pass forward through all features, returns logits



        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
