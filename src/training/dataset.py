# PYTORCH DATASET DEFINITION

# Wraps the preprocessed numpy array (X_train.npy, y_train.npy, etc.) in a PyTorch "Dataset" object.
# Why? -> PyTorch's training tools (specifically the DataLoader for Batches used in train_cnn.py) expects data to come from an object that follows:
    # __len__(self): returns how many examples the datset has
    # __getitem__(self, idx): returns the (input, labels) pair for a single example at position 'idx'

# By writing own small class that follows this interface, the DataLoader can automatically handle:
    # shuffling the data each epoch
    # splitting it into "batches" (small groups processed together)
    # (and later if needed: loading data in parallel using multiple workers)

import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    # Each item in this dataset is one heartbeat (1D signal, 187 datapoints -> the ECG waveform & an integer label per class)
    # PyTorch's Conv1D layers expect input shaped: (batch_size, num_channels, sequence_length)
        # num_channels = 1 (single ECG channel) - unlike e.g. RGB images with 3 channels - so each signal need shape (1, 187).

    def __init__(self, X, y):
        # Parameters: 
            # X -> numpy array, shape (num_samples, 187) - ECG signal
            # y -> numpy array, shape (num_samples,)     - the integer labels

        # torch.from_numpy() -> converts numpy array into a PyTorch "tensor" - PyTorch's version of a numpy array. Behaves similarly, but can additionally:
            # - be moved to a GPU for faster computation
            # - automatically track gradients for backpropagation (which is handled internally by PyTorch's "autograd" system)
                # "Backward propagation of errors" = math algorithm to efficiently compute the gradient of a neural network's loss function. 

                # Gradient = a list of numbers telling 2 things: 
                # 1. Which direction increases the error the fastest
                # 2. How steep that slope is
                # If the model has 100 weights, the gradient is a list of 100 specific numbers - one for each weight
                # Each number inside the gradient tells you the relationship between a specific weight and the total error.
                # Example: if the gradient for weight_A is 3.5 it means: if you increase weight_A slightly, the total error will go up quickly.
                # To lower the error, your code needs to decrease weight_A
                # Similarily: If the gradient for weight_B is -0.1, it means: if you increase weight_B, the total error will go down slightly
                # Subsequently: Because the slope is small (0.1 vs 3.5), changing this weight has a much smaller impact on the error.
                # Pseudo Code:
                # when calculating: loss.backward() -> calculating the gradient
                # now every weight as a .grad attribute holding specific slope number: print(model.weight.grad) - output might look like: tensor([0.142, -3.512, 0.002])
 
                # Backpropagation is done by systematically applying the "chain rule of calculus", working backards from the output layer to the input layer.
                # The chain rule is a math shortcut used to find out how a change at the beginning of a chain of events affects the final result.
                # Imagine three gears connected in a row: Gear A turns Gear B, and Gear B turns Gear C. If turning Gear A by 1 rotation makes Gear B turn 3 times (3 ×).
                # If turning Gear B by 1 rotation makes Gear C turn 2 times (2 ×). How much does Gear C turn if you rotate Gear A? You multiply them: 3 × 2 = 6 times.
                # In a neural network, these gears are layers. The chain rule lets you multiply the rates of change across layers to find out exactly how much 
                # changing a weight at the very beginning alters the error at the very end.
                
                # To understand: the Forward Pass = Network makes a prediction by passing data from the input to the output layer. 
                # The predicted output is then compared to known actual target --> gives the overall "loss" (the error)
                # Backward Pass = algorithm takes error and sends it backwards through the network (by applying the chain rule). It calculates how much
                # each weight and bias in the network contributed to the final error -> calculates the "exact numerical error blame"
                # Backpropagation saves and reused intermediate gradient calculations from later layers -> allowing for optimization techniques like Gradient Descent.

                # Gradient Descent then updates the weights -> takes the gradients calculated by the backpropagation and uses them to 
                # change the weights in the code (e.g., by substracting a fraction of the gradient from the current weight) -> reduces error
                # Gradient Descent looks at the numbers and updates weights like: New Weight = Old Weight - (Learning Rate * Gradient)
                # After Gradient Descent, the calculated gradients need to be cleared so that they dont add up in the next loop -> .zero_grad()

            # from_numpy() shares memory with the original array (efficient, no copy) and preserves dtype, intended, as int64 labels are what PyTorch's
            # CrossEntropyLoss expects

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        # this function will be called by the DataLoader to find out how many examples exist (to know how many batches make up one epoch)
        return len(self.y)
    
    def __getitem__(self, idx):
        # also called by the DataLoader once per example, with 'idx' being the position to retrieve (0,1,2,..., up to len(self)-1).
        # must return a tuple: (input_tensor, label_tensor)

        # self.X[idx] has shape (187,) -> 1D tensor with 187 values
        # .unsqueeze(0) -> adds a new dimension at position 0 -> turns the shape into (1, 187) => "1 channel, 187 timesteps" (needed for Conv1D)
        # the DataLoader later stacks many of these tensors into a batch of shape (batch_size, 1, 187)

        # self.y[idx] is a single integer (e.g. tensor(2) - no reshaping needed)

        signal = self.X[idx].unsqueeze(0)
        label = self.y[idx]

        return signal, label
    

class ECGAutoencoderDataset(Dataset):

    # PyTorch Dataset for Autoencoder training/ evaluation

    # Key difference vs. ECGDataset (that we use for the CNN)
        # ECGDataset returns signal,label-pairs - and signal shaped (1,187) for the Conv1d layers
        # Here arent any labels in the traditional sense -> autoencoder's target is the input itself.
        # --> __getitem__ only returns ONE tensor per example, not a tuple
        # the autoencoder (models.py) uses nn.Linear layers, which expect input shaped (batch, features) - i.e. (batch, 187), WITHOUT
        # an extra channel dimension -> here we dont call .unsqueeze(0)


    def __init__(self, X):
        # same conversion as in the ECGDataset: numpy array -> PyTorch tensor, preserving dtype float 32
        self.X = torch.from_numpy(X)    # (num_samples, 187)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]          # returns a single 1D tensor of shape (187,) -> which the DataLoader will stack many of these into a batch (batch_size, 187)
                                    # needed for the nn.Linear-based autoencoder 