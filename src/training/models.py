# 1D CONVOLUTIONAL NEURAL NETWORK (CNN) MODEL DEFINITION

# Defines the architecture of the NN for classifying ECG heartbeats

# Key building blocks used:

# nn.Conv1d
    # Slides small "filters" along the time axis to detect local patterns (e.g., sharp spikes), regardeless of WHEN in the 187-value window they occur
    # Key parameters:
        # in_channels: how many channels the input has (1 here)
        # out_channels: how many DIFFERENT filters this layers learns. Each filter can detect a different pattern. 
                      # More filters = more detectable patterns, but also more parameters to train
        # kernel_size: how many consecutive timesteps each filter looks at once (its "width")
        # padding: zero-values added at start/end of the input before filtering. Padding is used so the output length equals the input length
                      # for odd kernel_size, the formul is: padding = (kernel_size - 1) // 2

# nn.ReLU
    # An activation function. Replaces all negative values with 0 (keeps positives values unchanged). Without activation functions, stacking multiple layers would 
    # mathematically collapse into a single linear operation. Activation functions are what let NN learn complex, non-linear patterns.

    # Example: If you do not use activation functions, stacking 100 layers is mathematically identical to having just one single layer.
    # Here is the simple algebra why: Layer 1 does: Y = 2 * X. Layer 2 does: Z = 3 * Y.
    # If you stack them together without an activation function in between, Layer 2 just processes Layer 1's output: Z = 3 * (2 * X) = 6 * X

    # So without the activaton function, the whole network collapses into a basic linear regression model.

    # What is an activation function: a simple mathematical function applied to the output of a neuron before passing it to the next layer. 
    # Without it, a neuron only does basic math -> it multiplies inputs by weights and adds a bias (z = w * x + b) -> which is a 
    # strictly linear operation (a straight line graph).
    # The activation function breaks the linear chain so the math cannot be simplified or compressed.
    # The activation function (like ReLU or Sigmoid) takes that straight-line result and bends it, introducing a twist or a curve.

    # ReLU = Rectified Linear Unit -> if number is negarive, turn it to 0, if positiv, keep it.
    # Exmaple again: Layer 1: Y = 2 * X -> Apply ReLU -> Layer 2: Z = 3 * Y.
    # If X = -2 -> Layer 1 outputs -4 -> ReLU triggers: -4 is negative, so it becomes 0.
    # Layer 2 receives 0 and outputs 3 * 0 = 0 
    # If you didn't have ReLU, the math would have been 3 * (2 * -2) = -12\). Because 0 =! -12, the layers can no longer be compressed into a single formula. 
    # They must remain separate operations.

    # Analogy: Real world is not made of straight lines - but highly complex, chaotic curves. Activation function bends the stright lines at every layer
    # -> Deep Neural Network acts like a mathematical origami artist. 
    # Layer 1 creates simple straight cuts or lines
    # Layer 2 takes those lines, bends them at the activation functions and combines them into simple shapes (like triangles or squares)
    # Layer 3 takes those bent shapes, twists them again and combines them into complex curves

    # By stacking hundreds of layers, each separated by a non-linear bend - the network can approximate absolutely any complex shape, curve, or pattern in existence. 
    # This ability is known in data science as the Universal Approximation Theorem.

# nn.MaxPool1d:
    # = downsampling. 
    # MaxPool1d looks at a small group of consecutive values (e.g., a group of 2) and keeps only the maximum from each group, discarding the rest
        # 1. Reduces sequence length (faster, less overfitting)
        # 2. Makes network slightly more rubst to small shifts in where a pattern occurs

# nn.Flatten:
    # takes multi-dimensional tensor - e.g., shape (batch, channels, length)
    # then reshapes it into 2D - shape (batch, channels * length)
    # -> flattens everything except the batch dimension into 1 long vector per example
    # needed bc: nn.Linear expects a flat feature vector per example

# nn.Linear:
    # a "fully connected" Layer: every input value connects to every output value -> via a learnable weight
    # nn.Linear is used in the end to turn extracted features into final class scores

# nn.Dropout:
    # regularization technique - active only during training.
    # has probability p - and randomly sets some values to zero on each forward pass (different random values each time)
    # This prevents the network from relying too heavily on any single feature/ neuron, therefore reducing overfitting
    # During evaluation (model.eval()) -> Dropout is automatically disabled


import torch.nn as nn


class ECGCNNClassifier(nn.Module):
    # a small 1D CNN for classifying ECG heartbeats into 5 classes

    # Architecture Overview:
        # Layer ------> (new tensor shape)

        # Input:        (batch, 1, 187)
        # Conv1d + ReLU (batch, 16, 187)    <- 16 filters, same length (padding) - like having "16 channels", with 187 values.
        # MaxPool1d(2)  (batch, 16, 93)     <- length roughly halved
        # Conv1d + ReLU (batch, 32, 93)     <- 32 filters
        # MaxPool1d(2)  (batch, 32, 46)     <- length roughly halved - like having "32 channels", with 46 values.
        # Flatten       (batch, 1472)       <- 32*46 

        # to get from Flatten -> Linear + ReLu: input for next layer = (batch, 1472) ; Weight is a matrix with form (64, 1472) (a learnable parameter of the 
        # new layer) ; bias is a vector with form (64,) (also learnable parameter)

        # Why 64? --> this is a freely chosen design decision (a hyperparameter) - same as 16 and 32 amounts of filters in the Conv-Layers
        # Its "just" how much the 1472 (lots of probably redundant raw features) are compressed down to a smaller number of compact, more abstract features.
        # Out of these, the final layer makes a decision on categorizing in one of the 5 classes.
        # Its totally variable to decide to instead use e.g., 32, 128 or 100 - 64 is just a common integer that turned out to be practical to use 
        # It sits at a reasonable trade-off between model capacity, overfitting risk and computational effort.

        # Linear Layer output = input @ weight^T + bias 
        # -> every one of the 64 output values of the linear layr is calculated as a weighted sum of all 1472 input values + their own bias values
        # so for the output neuron "j": output[j] = sum(input[i] * weight[j][i] for i in range(1472)) + bias[j]
        # this happens 64 times (0-63), each time with different weights - each of the 64 output neurons looks on all 1472 inputs, but weights it differently
        # How many parameters does this layer have? --> 1472 * 64 (weights) + 64 (bias) = 94272 parameters
        # => the Linear Layer needs its own parameters for each Input-Output combination

        # Then ReLu Layer again: activation function: max(0,x) on all of the 64 values -> transforms the 64 raw linear combinations into non-linear activations.

        # Linear + ReLu (batch, 64)         <- "Feature-Extraction/Conv-Layer => Classification-Logic/ Fully-Connected-Layer"

        # Dropout       (batch, 64)         <- only active during training - randomly sets a selection of these 64 values to 0, e.g., with probability 0.5 half each run
                                               # similar to the linear layer -> input (batch,64) -> weight has shape (5,64) of learnable weights vector 
                                               # -> bias is (5,) of learnable bias vector --> again, each of the 5 defined output values (chosen bc we have 5 classes)
                                               # is a weighted sum of all 64 values from the previous layer + their own bias
                                               # amount of parameters = 64 * 5 (weights) + 5 (bias) = 325 parameter

        # Linear        (batch, 5)          <- 5 raw scores, 1 per class - often called "Logits"

        # These raw scores are not yet probabilities. Its just some positive or negative numbers.
        # The class with the highest score is the final prediction of the model.
        # For prediction: "argmax" over the 5 values -> index of class with highest score
        # For probability (later): "softmax" over the 5 Logits -> transform to values between 0 and 1 - with total sum of 1
        # During training: PyTorch's "CrossEntropyLoss" function expects exactly these raw Logits and then internally computes the softmax and loss 

    def __init__(self, num_classes=5):

        # nn.Module.__init__() must be called first - sets up internal bookkeeping that PyTorch needs (automatically find all parameters, layers, etc.)
        super().__init__()

        # First Convolutional block
        # 1 channel, 16 filters, kernel_size=5 => each filters looks at 5 consecutive timesteps, padding=2 => keeps output length == input length (187 -> 187)
        # since padding = (5-1) // 2 = 2
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()

        # MaxPool(kernel_size=2) -> looks at every 2 consecutive values, then keeps only max - roughly halving length
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second Convolutional block
        # 16 input channels to match first conv block output, 32 filters, same kernel and padding
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()

        # MaxPooling again with kernel_size=2
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Flatten + Fully connected Classification head
        # after pool2 -> tensor has shape (32,64) -> flatten turns into (batch, 32*64 = 1472)
        self.flatten = nn.Flatten()

        # Linear(1472,64): compresses the 1472 extracted features down to 64
        self.fc1 = nn.Linear(in_features=32*46, out_features=64)        # fc = fully connected / Dense Layer - every input neuron is connected with every output neuron
        self.relu3 = nn.ReLU()

        # Dropout(0.3): during training, randomly zero out 30% of the 64 values from fc1 on each forward pass.
        # 1 forward pass is = data fully running through the whole network 1 time.
        # Typical Training-Loop: 
            # 1. Forward Pass: Batch is sent through network -> predictions made (y_pred is calculated)
            # 2. Calculatin Loss: Comparing predictions with true labels
            # 3. Backward Pass: Calculating gradients (-> how much each weight needs to be adjusted)
            # 4. Optimization Step: Weights are updated/ recalculated
        # Model sees lots of Batches during training -> therefore also many forward passes.
        # Example: With 87554 training samples and batch size of 64 -> 1369 forward passes
        # -> example: in Batch 1 the neurons 1,4,7,25,... are set to 0. In Batch 2, the neurons 2,3,6,23,... are set to zero, etc.
        self.dropout = nn.Dropout(p=0.3)

        # Final Layer: Linear(64, num_classes) -> produces 1 raw score/ logit per class.
        # no softmax is applied here, as loss function expects raw, un-normalized scores and applies necessary math internally
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)  

    # define "Forward Pass" function that is later called when training the CNN in "train_cnn.py"
    def forward(self, x):
        # Defines how data flows through the network. 
        # PyTorch calls this automatically when you do 'model(some_input)'

        # Parameter:
            # x -> tensor of shape (batch_size, 1, 187) - a batch of ECG signals (1 channel, 187 timesteps) - x is the input that is given to this function during training
        
        # Returns:
            # tensor of shape (batch_size, num_classes) - raw class scores/ logits for each example in the batch

        x = self.conv1(x)   # (batch, 16, 187)
        x = self.relu1(x)   # (batch, 16, 187)  - negative values -> 0
        x = self.pool1(x)   # (batch, 16, 93)

        x = self.conv2(x)   # (batch, 32, 93)
        x = self.relu2(x)   # (batch, 32, 93)
        x = self.pool2(x)   # (batch, 32, 46)

        x = self.flatten(x)  # (batch, 1472)

        x = self.fc1(x)      # (batch, 64)
        x = self.relu3(x)    # (batch, 64)
        x = self.dropout(x)  # (batch, 64)  - some values randomly zeroed during training

        x = self.fc2(x)      # (batch, 5)   - final class scores (logits)
    
        return x
    


class ECGAutoencoder(nn.Module):

    # A simple "dense" (fully-connected) Autoencoder for ECG heartbeats.

    # Unlike the CNN, this NN has no convoluational layers -> therefore: every value connects to every value of the next layer via nn.Linear layers.
    # Standard, simple choice for fixed-length inputs like the 187-value signals
    # Avoids extra complexity of "transposed convolutions" that a conv1d-based decoder would require

    # Architecture: for a single example, batch dimension ommited for clarity

    # Input:        (187)

    # -- Encoder --
    # Linear (187 -> 64) + ReLu
    # Linear (64  -> 16) + ReLu     <- this is the bottleneck / latent representation: just 16 numbers summarizing the 187-value input

    # -- Decoder --
    # Linear (16 ->  64) + ReLu     <- linear matrix operation that guarantees exactly 64 output values
    # Linear (64 -> 187) + Sigmoid  <- linear matrix operation that guarantees exactly 187 output values - exactly what we want

    # Output:       (187)           <- the Reconstruction, values squeezed into (0,1) by sigmoid

    # nn.Sequential is a convenient PyTorch container: it chains a list of layers together, automatically passing the output of each layer 
    # as the input of the next -> simpler than writing out each step in "forward()" individually (as done above for the CNN)
    # nn.Sequential is a natural fit when the data flows through layers in a simple, straight line (no branching)

    def __init__(self, input_dim=187, bottleneck_dim=16):
        super().__init__()

        # Encoder
        # takes the 187-value input and compresses it down to bottleneck_dim
        # using 64 as an intermediate stepping stone - gives the network more capacity to leanr useful intermediate features (in theory possible to go straight from 187 -> 16)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    
    def forward(self, x):
        # Parameter: x -> tensor of shape (batch_size, 187)
        # Returns: Reconstruction -> tensor of shape (batch_size, 187) => model's attempt to reproduce x out of the downsampled representation

        # Important: not returning the bottleneck representation 'z' here.
        # For this project: only the final reconstruction is returned -> to compute the reconstruction error. 
        # However, in other applications, like dimensionality reduction/ compression or visualization it might be exactly the goal to return 'z'.

        z = self.encoder(x)                 # (batch, 16) -> the compressed version
        reconstruction = self.decoder(z)    # (batch, 187) -> the attempt to revuild input

        return reconstruction

