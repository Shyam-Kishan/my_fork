import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage
import torch
from torch import nn
import numpy
torch.__version__

K_P = 1
K_I = 0.4
K_D = 0.2

STEPS = 550

# Instantiate Car
car = make_car(desired_v=20.0, dt=0.1)

# Initialize lists to store data
velocities = []
errors = []
times = []

# 1/10th of a second has passed per iteration
for i in range(STEPS):
    print(f"===== Step : {i} =====")
    calculate_desired_acceleration(car=car, K_P=K_P, K_I=K_I, K_D=K_D)
    throttle_percentage = acceleration_to_throttle_percentage(acceleration_desired=car["desired_a"])
    update(car=car, throttle_perc=throttle_percentage)

    # Add data into lists
    velocities.append(car["v"])
    errors.append(car["error_prev"])
    times.append(car["t"])

### Graphs

# Figure 1 - Velocity Over Time
plt.figure(1)
plt.plot(times, velocities, color="blue", linestyle="--", marker="o")
plt.title("Velocity over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")

# Figure 2 - Error Over Time
plt.figure(2)
plt.plot(times, errors, color="red", linestyle="--", marker="o")
plt.title("Error over Time")
plt.xlabel("Time (s)")
plt.ylabel("Errors")
plt.show()