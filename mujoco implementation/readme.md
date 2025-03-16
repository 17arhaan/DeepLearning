# Assistive Motion Simulation Using MuJoCo

This repository contains a basic humanoid simulation using MuJoCo's Python bindings. The project is dedicated to research in assistive motion, aimed at enhancing mobility for individuals with disabilities. This simulation provides a foundational framework for exploring gait control and developing assistive strategies in a virtual environment.

## Repository Structure

```
17arhaan/
├── DeepLearning/
├── mujoco_implementation/
│   ├── humanoid.xml        # MuJoCo humanoid model file
│   ├── view.py             # Simulation and visualization script
│   ├── LICENSE             # MIT License file
│   ├── README.md           # This file
```

## Requirements

Ensure you have Python 3 installed, then install dependencies using:

```sh
pip install mujoco numpy
```

## Usage

1. Ensure that `humanoid.xml` is located in the same directory as `view.py`.
2. Run the simulation with:

   ```sh
   python view.py
   ```

A viewer window will open, displaying the humanoid model. The simulation applies periodic control inputs to emulate rudimentary walking motion—a basis for research in assistive mobility.

## Code Overview

### `view.py`

- Loads the humanoid model and initializes the simulation state.
- Sets up a real-time viewer to visualize the simulation.
- Applies simple periodic control signals (sine waves) to selected actuators to stimulate walking-like motion.
- Serves as a preliminary framework for developing advanced assistive motion controllers.

## Customization

### Actuator Indices
Modify actuator indices in `view.py` to match your model’s configuration. Example:

```python
left_hip = 0
right_hip = 1
```

### Control Parameters
Fine-tune gait patterns by adjusting frequency, amplitude, and phase offsets:

```python
amplitude = 0.5  # Adjust step height
frequency = 2.0  # Adjust walking speed
phase_offset = 0.0  # Synchronization of movements
```

Experimenting with these values can help simulate different walking styles or rehabilitation strategies.

## Acknowledgments

This project is committed to advancing research in assistive technology for individuals with disabilities. Contributions and feedback are welcome to improve this framework and better support mobility and rehabilitation research.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

