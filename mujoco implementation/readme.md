# Assistive Motion Simulation Using MuJoCo

This repository contains a basic humanoid simulation using MuJoCo's Python bindings. The work presented here is dedicated to research in assistive motion—aimed at stimulating movement and improving mobility in individuals with disabilities. The simulation provides a starting framework for exploring gait control and other assistive strategies in a virtual environment.

## Repository Structure
  ```
17arhaan/
└── DeepLearning/
└── mujoco implementation/
├── humanoid.xml        # MuJoCo humanoid model file
├── view.py             # Simulation and visualization script
├── LICENSE             # MIT License file
└── README.md           # This file
  ```

## Requirements

- **Python 3.x**
- **MuJoCo Engine and Python Bindings**  
  ```
  pip install mujoco
  ```
-	**NumPy**
  ```
pip install numpy
  ```
Usage
	1.	Ensure that the humanoid.xml file is located in the same directory as view.py.
	2.	Run the simulation with:
python view.py


A viewer window will open, displaying the humanoid model. The simulation applies periodic control inputs to emulate a rudimentary walking motion—a basis for research in assistive motion for individuals with disabilities.

Code Overview
	•	view.py
	•	Loads the humanoid model and initializes the simulation state.
	•	Sets up a real-time viewer to visualize the simulation.
	•	Applies simple periodic control signals (sine waves) to selected actuators to stimulate walking-like motion.
	•	Serves as a preliminary framework for exploring and developing advanced assistive motion controllers.

Customization
	•	Actuator Indices:
Adjust the actuator indices in view.py (e.g., left_hip and right_hip) to match your model’s configuration.
	•	Control Parameters:
Modify the frequency, amplitude, and phase offsets of the control signals to fine-tune the gait or explore other motion patterns.

Acknowledgments

This project is committed to advancing research in assistive technology for individuals with disabilities. Your feedback and contributions are welcome as we refine this framework to better support mobility and rehabilitation research.

License

This project is licensed under the MIT License.
