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
K_I = 0.0
K_D = 0.0
 
STEPS = 550

car0 = make_car(desired_v=5.0, dt=0.1)
car1 = make_car(desired_v=10.0, dt=0.1)
car2 = make_car(desired_v=15.0, dt=0.1)
car3 = make_car(desired_v=20.0, dt=0.1)
car4 = make_car(desired_v=25.0, dt=0.1)
car5 = make_car(desired_v=30.0, dt=0.1)
car6 = make_car(desired_v=35.0, dt=0.1)
car7 = make_car(desired_v=40.0, dt=0.1)
car8 = make_car(desired_v=22.0, dt=0.1)
car9 = make_car(desired_v=33.0, dt=0.1)


accelerations = []          # Target predicton for model to make    (output)
velocities = []             # Model takes this to predict accel     (input)
desired_v = []              # Model takes this to predict accel     (input)
errors = []
times = []

# Function to generate PID data
def create_car_data(car_x: dict):
    for i in range (STEPS):
        print(f"For Desired Velocity: {car_x["desired_v"]}")
        print(f"===== Step : {i} =====")
        calculate_desired_acceleration(car=car_x, K_P=K_P, K_I=K_I, K_D=K_D)
        throttle_percentage = acceleration_to_throttle_percentage(acceleration_desired=car_x["desired_a"])
        update(car=car_x, throttle_perc=throttle_percentage)

        velocities.append(car_x["v"])
        desired_v.append(car_x["desired_v"])
        accelerations.append(car_x["desired_a"])
        errors.append(car_x["error_prev"])
        times.append(car_x["t"])


### Creating data using 10 different cars and 10 different speeds

# Car 1
create_car_data(car1)

# Car 2
create_car_data(car2)

# Car 3
create_car_data(car3)

# Car 4
create_car_data(car4)

# Car 5
create_car_data(car5)

# Car 6
create_car_data(car6)

# Car 7
create_car_data(car7)

# Car 8
create_car_data(car8)

# Car 9
create_car_data(car9)

### Create Data

# Gather Velocity data
X1 = torch.tensor(data=velocities, dtype=torch.float32).unsqueeze(dim=1)

# Gather Desired Velicty data
X2 = torch.tensor(data=desired_v, dtype=torch.float32).unsqueeze(dim=1)

# Combined Velicty adn Desired Velocity Data into 2-D tensor
X = torch.cat((X1, X2), dim=1)
# print(f"X shape: {X.shape}")

# Gather Desired Acceleration data
y = torch.tensor(data=accelerations).unsqueeze(dim=1)
# print(f"y shape: {y.shape}")


# Splitting Data - 75% of all data collected will be used for training
train_split = int(0.75 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Test to see if training split is correctly made
# print(len(X_train), len(y_train), len(X_test), len(y_test))

# Creating Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__()
        # Have in initialize a Linear Layer 
        self.linear_layer = nn.Linear(in_features=2, out_features=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
# Set the manual seed when creating the model (not always needed)
torch.manual_seed(42)
model_1 = LinearRegressionModel()

# Creating loss function
loss_fn = nn.L1Loss()

# Creating optimizer
# optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.0006625)
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.0009)


# Set the number of epochs
epochs = 1000

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_1.train()

    # 1. Forward pass on train data using the forward() method
    y_pred = model_1(X_train)

    # 2. Calculate the loss (how different are our models predictions to ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    ### Testing

    # Put model in evaluation (testing) mode
    model_1.eval()

    # 1. Forward pass on test data
    with torch.inference_mode():
        test_pred = model_1(X_test)

    # debug: Debugger says target size is different than input size for loss_fn()
    # print(len(test_pred), len(y_test))
    # This isn't the issure, test_pred and y_test are of same length for each epoch

    # 2. Calculate the loss
    test_loss = loss_fn(test_pred, y_test.type(torch.float))        # predictions come in torch.float datatype, so comparisons needs to be done with tensors of the same type

    # Print out what's happening
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test loss: {test_loss }")


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    '''
    Plots training data, test data, and compares predictions
    '''
    # plt.figure(figsize=(10, 7))

    ### Plotting Data - using cars' current velocities as X, desired accelerations as Y

    # Plot training data in blue
    plt.scatter(train_data[:, 0], train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data[:, 0], test_labels, c="g", s=5, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red
        plt.scatter(test_data[:, 0], predictions, c="r", s=6, label="Predictions")

    # Show the legend
    plt.xlabel("Current Velocity")
    plt.ylabel("Desired Aceeleration")
    plt.title("Linear Regression Model Predicting Desired Acceleration")
    plt.legend(prop={"size" : 14})
    plt.show()

model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)

# Show graphs
plot_predictions()
plot_predictions(predictions=y_preds)