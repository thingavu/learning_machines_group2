from coppeliasim_zmqremoteapi_client import RemoteAPIClient


client = RemoteAPIClient(
    host="localhost",  # This is just to say, "this same computer"
    port=23000,  # The default port CoppeliaSim launches the ZMQ API at.
)
# This is Lua nonsense. We are gathering a global Lua object called "sim",
# and calling python functions on *that*.
# You can ignore it, and pretend we created a "sim" object.
sim = client.require("sim")

# Start the simulation
sim.startSimulation()

# Wait for 5 second
sim.wait(5)

# Stop the simulation
sim.stopSimulation()