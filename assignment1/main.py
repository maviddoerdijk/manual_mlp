import numpy as np
import random

from mlp_numpy import MLP, CrossEntropyModule

np.random.seed(42)

# 1. buncha random data creation
n_inputs = 10 
n_hidden = 128
n_classes = 5 # say we are doing classification with 5 classes (since we'll have implemented softmax in the end)
batch_size = 4
x = np.random.randn(batch_size, n_inputs) # gives arr of shape (4, 10)
y_true = np.random.randint(0, n_classes, size=batch_size)
print(y_true)

# 2. create MLP and initialize a bunch of objects
# we want to have (one layer, 128 hidden units, 10 epochs, learning rate 0.1, seed 42)
mlp = MLP(n_inputs=n_inputs, n_hidden=n_hidden, n_classes=n_classes)
loss_fn = CrossEntropyModule()
epochs = 10
learning_rate = 0.1


# 3. do forward stuff
out = mlp.forward(x) 
y_pred_labels = np.argmax(out, axis=1)
loss = loss_fn.forward(out, y_true)
print(f"Calculated loss: {loss}")

# # 4. do backward stuff
dout = loss_fn.backward(out, y_true) # computes the very first gradient: dLoss/dout, or in more pedantic terms dL/dy_pred where y_pred is the output VECTOR of the network
mlp.backward(dout)

# # 5. update weights over epochs
