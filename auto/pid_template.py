import numpy as np

def make_car(desired_v:float=20.0, dt:float=0.1) -> dict:
    """ 
    Generates a dictionary that holds all the car's values. Keeps track of state varaibles.
    """
    car_state_dictionary : dict[str, float] = {
        "v" : 0, #velocity of your car 
        "a" : 0, #acceleration of your car
        "t" : 0, #time of your car
        "x" : 0, #position of your car
        "dt" : dt, #time step of your car, how much the time changes every time you update/step
        "desired_v" : desired_v, #desired velocity of your car, the velocity you want to maintain
        "step" : 0,
    
        #hint: use these variables in the integral and derivative portion of your PID control (steps 5 and 6 )
        "error" : None, 
        "error_prev" : 0.0,
        "net_integral" : 0.0,
        "desired_a" : 0.0
    }
    return car_state_dictionary

def update(car: dict, throttle_perc: float, mass: float = 1000, max_throttle_force: float = 5000, friction: float = 2.0) -> None:
        """
        Updates the car's state variables based on the throttle percentage.
        Use this function after finding throttle percentage to update the car's state variables.

        Inputs:
        car: dictionary containing the car's state variables
        throttle_perc: float, throttle percentage (-1 to 1)

        Outputs:
        None, but updates the car's state variables
        """
        force = throttle_perc * max_throttle_force
        car["a"] = (force / mass) - friction
        car["v"] += car["a"] * car["dt"]
        car["x"] += car["v"] * car["dt"]
        car["t"] += car["dt"]
        car["step"] += 1


def calculate_desired_acceleration(car: dict, K_P: float, K_I: float = 0.0, K_D: float = 0.0) -> tuple[float, float]:
        #input: car["v"], car["desired_v"] (floats)
        #output: desired acceleration and error tuple(float, float)

        # Calculating current error between desired_v and current_v 
        car["error"] = car["desired_v"] - car["v"]

        # Calculating total error amassed throughout time
        car["net_integral"] += car["error"]
        
        # Calculating desired_a using PID equations
        car["desired_a"] = (K_P * car["error"]) + (K_I * car["net_integral"] * car["dt"]) + (K_D * (car["error"] - car["error_prev"]) * car["dt"])

        # Storing previous error for Differntial implementation
        car["error_prev"] = car["error"]

        print(f"Error: {car['error']}")
        print(f"Current Velocity: {car["v"]} m/s")
        print(f"Acceleration Desired: {car["desired_a"]} m/s^2")

        return (car["desired_a"], car["error"])




def acceleration_to_throttle_percentage(acceleration_desired: float, mass: float = 1000, max_throttle_force: float = 5000) -> float:
        #input: desired_acceleration(float)
        #output: throttle percentage (float, -1 to 1)

        # a_max = F_max / total_mass
        acceleration_max = max_throttle_force / mass
        print(f"Max Acceleration: {acceleration_max} m/s^2")

        # throttle_percentage = a_desired / a_max
        throttle_percentage = acceleration_desired / acceleration_max                         # creating throttle percentage
        res = np.clip(a=throttle_percentage, a_min=-1.0, a_max=1.0)                           # clipping throttle percentage
        print(f"Throttle %: {res * 100}\n")

        return res