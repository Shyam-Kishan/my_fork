import torch
from torch import nn
import matplotlib.pyplot as plt
torch.__version__

# Scalar - a single number and it's a zero dimensino tensor
# scalar = torch.tensor(7)

# Get the Python number within a tensor (only works with one-element tensors)
# item = scalar.item()
# print(item)

#print(scalar.ndim)         Prints the number of dimensions the tensor has

# ====================================================================

# Vector - single dimension tensor but contains multiple numbers
# vector = torch.tensor([7, 7, 7])
# print(vector)
# print(vector.ndim)
# print(vector.shape)

#=================================================

# Matrix - a 2-dimensional array of numbers
# MATRIX = torch.tensor([[7, 8],
#                       [9, 10]])

# print(MATRIX)
# print(MATRIX.ndim)
# print(MATRIX.shape)

# ===================================================

# Tensor - an n-dimensional array of numbers
# TENSOR = torch.tensor([[[1, 2, 3], 
#                         [3, 6, 9], 
#                         [2, 4, 5]]])
# print(TENSOR)

# ==========================================

# Create a random tesnsor of size (3, 4)
# random_tensor = torch.rand(size=(3, 4))
# print(random_tensor, random_tensor.dtype)

# Createa a random tesnsor of size (244, 244, 3)
# random_tensor2 = torch.rand(size=(244, 244, 3))
# print(random_tensor2)

# ============================================================

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create Data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
# print(X[:10], y[:10])
# print(X, y)

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# print(len(X_train), len(y_train), len(X_test), len(y_test))

# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)

torch.manual_seed(42)

model1_0 = LinearRegressionModel()

print(list(model1_0.parameters()))

print(model1_0.state_dict())

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    '''
    Plots training data, test data, and compares predictions
    '''

    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size" : 14})

    plt.show()

plot_predictions()