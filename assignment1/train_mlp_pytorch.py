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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      llabels: 1D int array of size [batch_size]. Ground truth labels for
               each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the predicted class index by finding the max logit
    pred_labels = torch.argmax(predictions, dim=1) # use dim=1 to get max per row (find )
    correct = pred_labels.eq(targets).sum().item()
    
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
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    total_correct = 0
    total_samples = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.view(inputs.size(0), -1) # flatten from standard [B, 3, 32, 32] (CIFAR-10) to [B, 3*32*32]
            
            predictions = model(inputs)
            
            # get accuracy of the batch
            pred_labels = torch.argmax(predictions, dim=1)
            total_correct += pred_labels.eq(targets).sum().item()
            total_samples += len(targets)
            
        avg_accuracy = total_correct / total_samples # use total samples so that the average accuracy is independent of batch size

    #######################
    # END OF YOUR CODE    #
    #######################
    
    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO: actually use this

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    train_loader = cifar10_loader['train']
    val_loader = cifar10_loader['validation']
    test_loader = cifar10_loader['test']
    
    n_inputs = 32*32*3 # should be okay to hardcode, since it's common knowledge that CIFAR is RGB of 32x32 images
    n_classes = 10
    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=n_classes, use_batch_norm=use_batch_norm)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    val_accuracies = []
    logging_dict = {
        'train_accuracies': [],
        'train_losses': [],
        'val_accuracies': [],
        'val_losses': []
    }
    best_model = None # we need to return the model that worked best on dataset
    best_val_accuracy = 0.0

    # Training loop including validation
    for epoch in tqdm(range(epochs)):
      model.train()
      for train_inputs, labels in train_loader:
          # flatten inputs
          train_inputs = train_inputs.view(train_inputs.size(0), -1) # flatten from [B, 3, 32, 32] to [B, 3*32*32]
          optimizer.zero_grad() # reset gradients

          preds = model(train_inputs)
          loss = loss_fn(preds, labels)
          # train steps
          loss.backward() # compute gradients
          optimizer.step() # update weights (optimizer was given model.parameters())
          acc = accuracy(preds, labels)
          logging_dict['train_accuracies'].append(acc)
          logging_dict['train_losses'].append(loss.item())
      # avg_train_loss = epoch_loss / num_samples # seems like a weird way

      
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

    logging_dict['test_accuracy'] = test_accuracy
    logging_dict['val_accuracies'] = val_accuracies

    model = best_model
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
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
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
    