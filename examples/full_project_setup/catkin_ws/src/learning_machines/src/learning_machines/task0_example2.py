from robobo_interface import IRobobo, SimulationRobobo
from data_files import RESULTS_DIR
import json

# Define wheel speed and movement commands
WHEEL_SPEED = 50
MOVE_FORWARD = WHEEL_SPEED, WHEEL_SPEED, 800  # Move forward for 800ms
MOVE_BACKWARD = -WHEEL_SPEED, -WHEEL_SPEED, 1500  # Move backward for 800ms

# Explicit mapping of IR sensor indices
IR_SENSOR_MAPPING = {
    "FrontC": 4,
    "FrontL": 2,
    "FrontR": 3,
}

# Define thresholds for front sensors
IR_SENSOR_THRESHOLDS = {
    "FrontC": 200,  # Threshold for detecting the wall directly ahead
    "FrontL": 200,  # Threshold for detecting the wall on the left
    "FrontR": 200,  # Threshold for detecting the wall on the right
}

results = dict()

def touch_wall_and_reverse(rob: IRobobo, steps: int = 15):
    """
    Robot moves forward until it detects a wall using front sensors.
    Once a wall is detected, it moves backward and resumes moving forward.
    """
    rob.play_simulation()  # Start simulation

    try:
        for step in range(steps):
            print(f"Step {step + 1}")

            # Read IR sensors
            ir_data = rob.read_irs()

            # Log sensor readings
            print("Sensor readings:", ir_data)
            results[step] = ir_data

            # Check front sensors for wall detection
            wall_detected = False
            for sensor_name in ["FrontC", "FrontL", "FrontR"]:
                sensor_index = IR_SENSOR_MAPPING[sensor_name]
                sensor_value = ir_data[sensor_index]
                threshold = IR_SENSOR_THRESHOLDS[sensor_name]

                if sensor_value > threshold:
                    print(f"Wall detected by {sensor_name}! Sensor value: {sensor_value}, Threshold: {threshold}")
                    wall_detected = True
                    break

            if wall_detected:
                # Stop, move backward, and resume forward movement
                rob.move(0, 0, 500)  # Stop
                rob.sleep(0.5)
                print("Moving backward to avoid the wall.")
                rob.move(*MOVE_BACKWARD)
                rob.move(*MOVE_BACKWARD)
                rob.sleep(0.5)
                break
            else:
                # Continue moving forward
                print("No wall detected. Moving forward.")
                rob.move(*MOVE_FORWARD)
                rob.sleep(0.5)

    finally:
        rob.stop_simulation()  # Stop the simulation after completion
        print("Task completed.")
    
    with open(RESULTS_DIR / "sim_task0_example2.json", "w") as outfile:
        json.dump(results, outfile, indent=4, sort_keys=False)    

def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    touch_wall_and_reverse(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
