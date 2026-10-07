import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage
import torch
from torch import nn
import numpy

torch.__version__


K_P = 1
K_I = 0.4           # 0.4
K_D = 0.2           # 0.2
 
STEPS = 550
 
car = make_car(desired_v=20.0, dt=0.1)

velocities = []
errors = []
times = []

for i in range(STEPS):
    print(f"===== Step : {i} =====")
    calculate_desired_acceleration(car=car, K_P=K_P, K_I=K_I, K_D=K_D)
    throttle_percentage = acceleration_to_throttle_percentage(acceleration_desired=car["desired_a"])
    update(car=car, throttle_perc=throttle_percentage)

    velocities.append(car["v"])
    errors.append(car["error_prev"])
    times.append(car["t"])

# Create known parameters
weight = 0.7
bias = 0.3

# Create Data
start = 0
end = 1
step = 0.02
# X = torch.arrange(start, end, step).unsqueeze(dim=1)
X = torch.tensor(data=velocities)
X = X.unsqueeze(dim=1)

y = weight * X + bias

# Splitting Data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Test to see if training split is correctly made
# print(len(X_train), len(y_train), len(X_test), len(y_test))

# Creating Linear Regression model class
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
    
# Set the manual seed when creating the model (not always needed)
torch.manual_seed(42)
model_1 = LinearRegressionModel()
# print (mode1_1, mode1_1.state_dict())

# Creating loss function
loss_fn = nn.L1Loss()

# Creating optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)



plt.plot(times, velocities, color="blue", linestyle="--", marker="o")
plt.title("Velocity over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.show()

plt.plot(times, errors, color="red", linestyle="--", marker="o")
plt.title("Error over Time")
plt.xlabel("Time (s)")
plt.ylabel("Errors")
plt.show()