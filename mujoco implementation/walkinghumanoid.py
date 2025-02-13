import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the humanoid model.
model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# Launch the viewer.
viewer = mujoco.viewer.launch(model, data)

# --- Define actuator indices (example values) ---
# Replace these indices with the correct ones from your humanoid model.
# For demonstration, we assume:
#   - Left hip actuators are at indices 2 and 3.
#   - Right hip actuators are at indices 4 and 5.
# Similarly, you could define indices for knees and ankles.
left_hip_indices = [2, 3]
right_hip_indices = [4, 5]
# (Optional) Define knee and ankle indices if available.
# left_knee_indices = [6, 7]
# right_knee_indices = [8, 9]
# left_ankle_indices = [10]
# right_ankle_indices = [11]

# --- Define control parameters ---
freq = 1.5          # Frequency in Hz.
amp_hip = 0.4       # Amplitude for hip joints.
# amp_knee = 0.5    # Amplitude for knee joints.
# amp_ankle = 0.3   # Amplitude for ankle joints.

while viewer.is_running():
    t = data.time
    data.ctrl[:] = 0  # Reset all controls to zero.

    # Apply control to hip actuators:
    for i in left_hip_indices:
        data.ctrl[i] = amp_hip * np.sin(2 * np.pi * freq * t)
    for i in right_hip_indices:
        # Phase shift by pi for alternating motion.
        data.ctrl[i] = amp_hip * np.sin(2 * np.pi * freq * t + np.pi)

    # If you have knee/ankle actuators, you can add similar controls:
    # for i in left_knee_indices:
    #     data.ctrl[i] = amp_knee * np.sin(2 * np.pi * freq * t)
    # for i in right_knee_indices:
    #     data.ctrl[i] = amp_knee * np.sin(2 * np.pi * freq * t + np.pi)
    #
    # for i in left_ankle_indices:
    #     data.ctrl[i] = amp_ankle * np.sin(2 * np.pi * freq * t)
    # for i in right_ankle_indices:
    #     data.ctrl[i] = amp_ankle * np.sin(2 * np.pi * freq * t + np.pi)

    # Step the simulation.
    mujoco.mj_step(model, data)
    
    # Render the current state.
    viewer.render()
    
    # Optional: slow down the loop for smoother visualization.
    time.sleep(0.01)
