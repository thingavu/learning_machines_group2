from robobo_interface import IRobobo, SimulationRobobo
from data_files import RESULTS_DIR
import json

# Define wheel speed and movement commands
WHEEL_SPEED = 50
MOVE_FORWARD = WHEEL_SPEED, WHEEL_SPEED, 800  # Move forward for 500ms
TURN_RIGHT = WHEEL_SPEED, -WHEEL_SPEED, 500

# Explicit mapping of IR sensor indices
IR_SENSOR_MAPPING = {
    "BackL": 0,
    "BackR": 1,
    "FrontL": 2,
    "FrontR": 3,
    "FrontC": 4,
    "FrontRR": 5,
    "BackC": 6,
    "FrontLL": 7,
}

# Define thresholds for each sensor
IR_SENSOR_THRESHOLDS = {
    "FrontC": 20,  # Front center
    "FrontL": 100,  # Front left
    "FrontR": 100,  # Front right
    "FrontLL": 50, # Front far-left
    "FrontRR": 50, # Front far-right
}

results = dict()

def avoid_obstacles(rob: IRobobo, steps: int = 15):
    """
    Robot moves forward until an obstacle is detected.
    If an obstacle is detected, it reacts appropriately and ensures that it clears the obstacle completely.
    """
    rob.play_simulation()  # Start simulation

    try:
        for step in range(steps):
            print(f"Step {step + 1}")

            # Read IR sensors
            ir_data = rob.read_irs()
            obstacle_detected = False

            # Log all sensor readings for debugging
            print("Sensor readings:", ir_data)
            results[step] = ir_data

            # Prioritize front sensors
            for sensor_name in ["FrontC", "FrontL", "FrontR", "FrontLL", "FrontRR"]:
                sensor_index = IR_SENSOR_MAPPING[sensor_name]
                sensor_value = ir_data[sensor_index]
                threshold = IR_SENSOR_THRESHOLDS[sensor_name]

                if sensor_value > threshold:
                    print(f"Obstacle detected in front by {sensor_name}!")
                    rob.move(0, 0, 500)  # Stop immediately
                    rob.sleep(0.5)  # Pause briefly
                    print("Turning right to avoid front obstacle.")
                    rob.move(*TURN_RIGHT)
                    rob.sleep(0.5)  # Pause briefly after turning
                    rob.move(*MOVE_FORWARD)  # Move forward to clear the obstacle
                    rob.sleep(0.5)
                    obstacle_detected = True
                    break

            # If no obstacles are detected, move forward
            if not obstacle_detected:
                print("No obstacles detected. Moving forward.")
                rob.move(*MOVE_FORWARD)
                rob.sleep(0.5)

    finally:
        rob.stop_simulation()  # Stop the simulation after completion
        print("Task completed.")
    
    with open(RESULTS_DIR / "sim_task0_example1.json", "w") as outfile:
        json.dump(results, outfile, indent=4, sort_keys=False)

def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo): 
        avoid_obstacles(rob)