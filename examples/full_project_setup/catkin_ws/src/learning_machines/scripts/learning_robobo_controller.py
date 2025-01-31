#!/usr/bin/env python3
import sys
from learning_machines import train_rl_model, test_rl_model, test_hardware
from robobo_interface import HardwareRobobo

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, specify:
            - `--hardware` or `--simulation` for robot mode.
            - `--train` or `--test` for RL mode."""
        )
    
    if sys.argv[1] == "--hardware":
        test_hardware()

    elif sys.argv[1] == "--simulation":
        if len(sys.argv) < 3:
            raise ValueError("Specify `--train` or `--test` after `--simulation`.")
        
        if sys.argv[2] == "--train":
            train_rl_model()
        elif sys.argv[2] == "--test":
            test_rl_model()
        else:
            raise ValueError(f"{sys.argv[2]} is not a valid option.")
    
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid option.")

# #!/usr/bin/env python3
# import sys

# from robobo_interface import SimulationRobobo, HardwareRobobo
# from learning_machines import run_all_actions


# if __name__ == "__main__":
#     # You can do better argument parsing than this!
#     if len(sys.argv) < 2:
#         raise ValueError(
#             """To run, we need to know if we are running on hardware of simulation
#             Pass `--hardware` or `--simulation` to specify."""
#         )
#     elif sys.argv[1] == "--hardware":
#         rob = HardwareRobobo(camera=True)
#     elif sys.argv[1] == "--simulation":
#         rob = SimulationRobobo()
#     else:
#         raise ValueError(f"{sys.argv[1]} is not a valid argument.")

#     run_all_actions(rob)
