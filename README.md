# Bike Power Simulator

A physics-based cycling power simulation with a graphical user interface. Calculate and visualize how bike speed relates to rider power output, weight, aerodynamics, and course conditions.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)

## Features

- **Physics Engine**: Accurate speed calculation from power using:
  - Aerodynamic drag (CdA based on rider position and height)
  - Rolling resistance (tire-type dependent)
  - Gravitational resistance (grade/slope)
  - Drivetrain efficiency losses
  - Air density variation with elevation and temperature

- **Interactive GUI**:
  - Real-time speed output as you adjust parameters
  - Rider inputs: power (watts), weight, height
  - Bike inputs: bike weight, tire type, riding position
  - Course inputs: gradient, wind speed, elevation, temperature
  - Speed vs. Power curve visualization
  - Power breakdown pie chart

- **Presets**: Quick-load common scenarios (flat road, climbing, time trial, etc.)

## Installation

### Requirements
- Python 3.10+
- tkinter (usually included with Python)
- matplotlib
- numpy

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-user>/bike-power-sim.git
cd bike-power-sim

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Usage

1. Launch the app with `python main.py`
2. Adjust rider, bike, and course parameters using the sliders and input fields
3. Click **Calculate** or enable **Auto-update** to see real-time results
4. View the Speed vs. Power curve and power breakdown chart
5. Use **Presets** to quickly load common riding scenarios

## Physics Model

The simulator solves for speed by balancing power input against resistive forces:

```
P = (F_aero + F_roll + F_gravity) × v / η

Where:
  F_aero   = 0.5 × ρ × CdA × (v + v_wind)² — aerodynamic drag
  F_roll   = Crr × m × g × cos(θ)           — rolling resistance
  F_gravity = m × g × sin(θ)                 — gravitational force
  η        = drivetrain efficiency (default 97%)
  ρ        = air density (adjusted for elevation & temperature)
```

Speed is found iteratively using Newton's method to solve this nonlinear equation.

## License

MIT License
