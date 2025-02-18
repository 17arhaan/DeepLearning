import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch(model, data)

freq = 1.5
amp = 0.4
left_hip = [2, 3]       # example indices for left leg actuators
right_hip = [4, 5]      # example indices for right leg actuators

while viewer.is_running():
    t = data.time
    data.ctrl[:] = 0
    for i in left_hip:
        data.ctrl[i] = amp * np.sin(2 * np.pi * freq * t)
    for i in right_hip:
        data.ctrl[i] = amp * np.sin(2 * np.pi * freq * t + np.pi)
    mujoco.mj_step(model, data)
    viewer.render()
    time.sleep(0.01)