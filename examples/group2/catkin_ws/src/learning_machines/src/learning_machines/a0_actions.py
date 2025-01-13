import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import datetime
import openpyxl
import time
import cv2
import re


from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)  # Show the emotion of the robot on the screen (For the simulation = printing)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)  # Let the robot make an emotion sound (For the simulation = printing)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)  # Set the led of the robot


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())  # Returns sensor readings:
    # [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
    image = rob.read_image_front()  # Get the image from the front camera as a numpy array in cv2 format
    cv2.imwrite(str(FIGURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())  # Get the current pan of the phone. Range: 0-100
    print("Phone tilt: ", rob.read_phone_tilt())  # Get the current tilt of the phone. Range: 26-109
    print("Current acceleration: ", rob.read_accel())  # Get the acceleration of the robot
    print("Current orientation: ", rob.read_orientation())  # Get the orientation of the robot


def test_move_and_wheel_reset(rob: IRobobo):
    # Move the robot wheels for `millis` time
    #         Arguments
    #         left_speed: speed of the left wheel. Range: -100-0-100. 0 is no movement, negative backward.
    #         right_speed: speed of the right wheel. Range: -100-0-100. 0 is no movement, negative backward.
    #         millis: how many milliseconds to move the robot
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())  # Get the wheel orientation and speed of the robot
    # Allows to reset the wheel encoder positions to 0.
    #         After calling this both encoders reset, making the current position the new reference
    #         in position 0.
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())


def test_sim(rob: SimulationRobobo):
    print("Current simulation time:", rob.get_sim_time())
    print("Is the simulation currently running? ", rob.is_running())
    rob.stop_simulation()
    print("Simulation time after stopping:", rob.get_sim_time())
    print("Is the simulation running after shutting down? ", rob.is_running())
    rob.play_simulation()
    print("Simulation time after starting again: ", rob.get_sim_time())
    print("Current robot position: ", rob.get_position())
    print("Current robot orientation: ", rob.get_orientation())

    pos = rob.get_position()
    orient = rob.get_orientation()
    rob.set_position(pos, orient)
    print("Position the same after setting to itself: ", pos == rob.get_position())
    print("Orient the same after setting to itself: ", orient == rob.get_orientation())


def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.phone_battery())
    print("Robot battery level: ", rob.robot_battery())


def test_phone_movement(rob: IRobobo):
    # Command the robot to move the smartphone holder in the horizontal (pan) axis.
    #         This function is synchronous.
    #
    #         Arguments
    #         pan_position: Angle to position the pan at. Range: 11-343.
    #         pan_speed: Movement speed for the pan mechanism. Range: 0-100.
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    # Command the robot to move the smartphone holder in the vertical (tilt) axis.
    #         This function is synchronous.
    #
    #         Arguments
    #         tilt_position: Angle to position the tilt at. Range: 26-109.
    #         tilt_speed: Movement speed for the tilt mechanism. Range: 0-100.
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def current_time(start_time):
    return round((time.time() - start_time) * 1000, 0)  # in milliseconds


def record_data(sensor_readings, timestamp, irs, event):
    print(f"IR data recorded at {timestamp}: {irs}")
    sensor_readings.append((timestamp, irs, event))
    return sensor_readings


def execute_task_0(rob: IRobobo, detection_threshold=30, speed=50, movement_duration=100):
    """Function to simulate task 0 for a single iteration,
    store sensor readings at regular intervals, and adjust behavior based on obstacles.
    """
    if isinstance(rob, SimulationRobobo):
        detection_threshold *= 5
    sensor_readings = []
    print("Starting task...")
    start_time = time.time()

    # Record initial state
    sensor_readings = record_data(sensor_readings, current_time(start_time), rob.read_irs(), event="Initial")

    # Start moving forward
    # [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
    while (rob.read_irs()[2] < detection_threshold  # FrontL
           and rob.read_irs()[3] < detection_threshold  # FrontR
           and rob.read_irs()[4] < detection_threshold  # FrontC
           and rob.read_irs()[5] < detection_threshold  # FrontRR
           and rob.read_irs()[7] < detection_threshold):  # FrontLL
        rob.move_blocking(speed, speed, movement_duration)
        sensor_readings = record_data(sensor_readings, current_time(start_time), rob.read_irs(), event="Forward")

    # When an object is reached:
    print("Object detected, moving backwards!")
    sensor_readings = record_data(sensor_readings, current_time(start_time), rob.read_irs(), event="Object detected")
    time.sleep(1)
    # for 2 seconds
    for steps in range(10):
        rob.move_blocking(-speed, -speed, movement_duration)  # movement_duration = 100 milliseconds
        sensor_readings = record_data(sensor_readings, current_time(start_time), rob.read_irs(), event="Backward")

    # Turn right (left wheel forward, right wheel backward)
    print("Turning right...")
    for steps in range(10):
        rob.move_blocking(speed, -speed, movement_duration)
        sensor_readings = record_data(sensor_readings, current_time(start_time), rob.read_irs(), event="Turning right")

    print("Going forward!")
    for steps in range(30):
        rob.move_blocking(speed, speed, movement_duration)
        sensor_readings = record_data(sensor_readings, current_time(start_time), rob.read_irs(), event="Forward")

    return sensor_readings


def store_sensor_data_in_excel(sensor_readings, experiment_number, iteration_number, rob: IRobobo):
    """Store sensor data in an Excel file, including time, sensor values, and experiment details.
    If the file already exists, append the new data to it.
    """
    # Create a DataFrame to store sensor readings along with additional info
    data = {
        "Time": [],
        "Event": [],
        "Sensor IR0": [],  # BackL
        "Sensor IR1": [],  # BackR
        "Sensor IR2": [],  # FrontL
        "Sensor IR3": [],  # FrontR
        "Sensor IR4": [],  # FrontC
        "Sensor IR5": [],  # FrontRR
        "Sensor IR6": [],  # BackC
        "Sensor IR7": [],  # FrontLL
        "Mode": [],
        "Iteration Number": []}

    for reading in sensor_readings:
        data["Time"].append(reading[0])  # timestamp
        data["Event"].append(reading[2])
        data["Mode"].append("Simulation" if isinstance(rob, SimulationRobobo) else "Hardware")
        data["Iteration Number"].append(iteration_number + 1)
        for ir, sensor_value in enumerate(reading[1]):  # reading[1] contains the IR sensor values
            data[f"Sensor IR{ir}"].append(sensor_value)

    new_data = pd.DataFrame(data)
    # Excel file path
    excel_path = FIGURES_DIR / f"sensor_data_experiment_{experiment_number}.xlsx"

    # Check if the file already exists
    if excel_path.exists():
        existing_data = pd.read_excel(excel_path, engine="openpyxl")
        # Concatenate existing data with the new data
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        # If file does not exist, only the new data will be saved
        combined_data = new_data

    # Save the data to the Excel file
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        combined_data.to_excel(writer, index=False)

    print(f"Sensor data saved to: {excel_path}")


def run_all_actions(rob: IRobobo, num_iterations: int = 1, experiment_number: int = 1):
    """Run the task with the robot for multiple iterations and save the data."""
    experiment_files = list(FIGURES_DIR.glob("sensor_data_experiment_*.xlsx"))
    if experiment_files:
        experiment_files.sort()
        latest_file = experiment_files[-1]
        match = re.search(r"experiment_(\d+)", latest_file.name)  # Extract the number after "experiment_"
        experiment_number = int(match.group(1)) + 1

    for iteration in range(num_iterations):
        pause = 3
        print(f"\nRunning iteration {iteration + 1} of {num_iterations}...")
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
            sensor_readings = execute_task_0(rob)
            rob.stop_simulation()
        else:
            pause = 15
            sensor_readings = execute_task_0(rob)

        store_sensor_data_in_excel(sensor_readings, experiment_number=experiment_number,
                                   iteration_number=iteration, rob=rob)

        if iteration < num_iterations - 1:
            print("The next iteration will start in...")
            for i in reversed(range(pause)):
                print(i + 1)
                time.sleep(1)
