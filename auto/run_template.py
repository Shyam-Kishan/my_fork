import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

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