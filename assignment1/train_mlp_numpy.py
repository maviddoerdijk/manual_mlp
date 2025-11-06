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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    pred_labels = np.argmax(predictions, axis=1)
    correct = np.equal(pred_labels, targets).sum().item()
    
    accuracy = correct / len(targets)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    total_correct = 0
    total_samples = 0

    for inputs, labels in data_loader:
        inputs = inputs.reshape(inputs.shape[0], -1) # flatten with np
        preds = model(inputs)

        # get accuracy of the batch
        pred_labels = np.argmax(preds, axis=1)
        total_correct += np.equal(pred_labels, labels).sum().item()
        total_samples += len(labels)

    avg_accuracy = total_correct / total_samples

    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    train_loader = cifar10_loader['train']
    val_loader = cifar10_loader['validation']
    test_loader = cifar10_loader['test']
    
    n_inputs = 32*32*3 # should be okay to hardcode, since it's common knowledge that CIFAR is RGB of 32x32 images
    n_classes = 10
    model = MLP(n_inputs = n_inputs, n_hidden=hidden_dims, n_classes=n_classes)
    loss_module = CrossEntropyModule()

    val_accuracies = []
    logging_dict = {
        'train_accuracies': [],
        'train_losses': [],
        'val_accuracies': [],
        'val_losses': []
    }
    best_model = None # we need to return the model that worked best on dataset
    best_val_accuracy = 0.0

    for epoch in tqdm(range(epochs)):
        for inputs, labels in train_loader:
            inputs = inputs.reshape(inputs.shape[0],-1) # [B, 3, 32, 32] -> [B, 3*32*32]
            preds = model(inputs) # call model directly because we implemented the __call__ function ourselves so that it works like pytorch
            loss = loss_module.forward(preds, labels) # softmax output -> CE loss

            acc = accuracy(preds, labels)
            logging_dict['train_accuracies'].append(acc)
            logging_dict['train_losses'].append(loss.item())

            # backprop
            loss_gradient = loss_module.backward(preds, labels) # gradient of loss w.r.t. softmax output
            _ = model.backward(loss_gradient) # should do backprop and update all weights accordingly
            # update gradients - update after entire batch
            model.lin_mod1.params['weight'] = model.lin_mod1.params['weight'] - lr * model.lin_mod1.grads['weight']
            model.lin_mod1.params['bias'] = model.lin_mod1.params['bias'] - lr * model.lin_mod1.grads['bias']
            model.lin_mod2.params['weight'] = model.lin_mod2.params['weight'] - lr * model.lin_mod2.grads['weight']
            model.lin_mod2.params['bias'] = model.lin_mod2.params['bias'] - lr * model.lin_mod2.grads['bias']
            
            # setting gradients back to zero was necessary in torch but should not be necessary here, since we do `self.grads['bias'] = ..` rather than +=
            model.clear_cache() # reset all cached values from feedforward

        # get validation accuracy
        val_accuracy = evaluate_model(model, val_loader)
        logging_dict['val_losses'].append(loss.item())
        val_accuracies.append(val_accuracy)
        print(f"\nValidation accuracy at epoch {epoch}: {val_accuracy}")

        if best_model is None or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model) # this is what the hint suggested

    print("Training finished. Evaluating on test set...")
    test_accuracy = evaluate_model(best_model, test_loader)
    print(f"Best Validation Accuracy achieved: {best_val_accuracy:.4f}")
    print(f"Test Accuracy of best model: {test_accuracy:.4f}")
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    