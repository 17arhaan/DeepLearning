import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the humanoid model from the XML file.
model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# Launch the viewer without the unsupported keyword.
viewer = mujoco.viewer.launch(model, data)

# Run the simulation loop while the viewer window is open.
while viewer.is_running():
    # Step the simulation forward.
    mujoco.mj_step(model, data)
    
    # Render the current state.
    viewer.render()
    
    # Optional: slow down the loop for smoother visualization.
    time.sleep(0.01)
