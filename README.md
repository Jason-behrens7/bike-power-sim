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
  - Wind direction modeling (headwind, crosswind, tailwind)
  - Wheel rotational inertia

- **Interactive Tabbed GUI**:
  - **Simulation tab**: Real-time speed calculation with sliders for all parameters
  - **Course Profile tab**: Define multi-segment routes with varying grades and see speed over distance
  - **Compare tab**: Build and compare multiple scenarios side-by-side with bar chart
  - **Export tab**: Export results, course profiles, and comparisons to CSV

- **Input Validation**: Real-time feedback for out-of-range parameter values

- **Unit Toggle**: Switch between metric (km/h, kg, m) and imperial (mph, lbs, ft)

- **Presets**: Quick-load common scenarios (flat road, climbing, time trial, etc.)
  - Save/load custom presets to JSON files

- **Course Profile Simulation**: Define multi-segment courses and simulate speed, time, and elevation across the entire route

- **Scenario Comparison**: Add multiple configurations side-by-side and compare results visually

- **CSV Export**: Export single results, course profiles, or scenario comparisons

## Installation

### Requirements
- Python 3.10+
- tkinter (usually included with Python; on macOS with Homebrew, install `python-tk`)
- matplotlib
- numpy

### Setup

```bash
# Clone the repository
git clone https://github.com/Jason-behrens7/bike-power-sim.git
cd bike-power-sim

# Create a virtual environment (recommended, required on macOS with Homebrew Python)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Usage

1. Launch the app with `python main.py`
2. **Simulation tab**: Adjust rider, bike, and course parameters using sliders; view real-time speed, power curve, and breakdown chart
3. **Course Profile tab**: Add segments with different grades/conditions, click "Run Profile" to see speed and elevation plots
4. **Compare tab**: Click "Add Current as Scenario" to snapshot the current settings, repeat with different configurations to compare
5. **Export tab**: Export results to CSV files
6. Use **Presets** to quickly load common riding scenarios, or save your own

## Physics Model

The simulator solves for speed by balancing power input against resistive forces:

```
P = (F_aero + F_roll + F_gravity) × v / η

Where:
  F_aero    = 0.5 × ρ × CdA × (v + v_wind_eff)² — aerodynamic drag
  F_roll    = Crr × m_eff × g × cos(θ)           — rolling resistance
  F_gravity = m_eff × g × sin(θ)                  — gravitational force
  v_wind_eff = wind_speed × cos(wind_direction)    — effective headwind component
  m_eff     = m × (1 + I_wheels / (m × r²))       — effective mass with wheel inertia
  η         = drivetrain efficiency (default 97%)
  ρ         = air density (adjusted for elevation & temperature)
```

Speed is found iteratively using Newton's method to solve this nonlinear equation.

## Running Tests

```bash
python -m unittest tests -v
```

## License

MIT License
