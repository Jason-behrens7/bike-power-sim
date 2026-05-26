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
  - Calorie & energy expenditure estimation

- **Interactive Tabbed GUI** (9 tabs):
  - **Simulation**: Real-time speed calculation with sliders, power zone display, calorie estimates, W/kg display
  - **Course Profile**: Multi-segment routes with GPX import, speed/elevation charts, animated ride playback
  - **Workout**: Interval/workout simulator with zone-colored power profiles
  - **Compare**: Side-by-side scenario comparison with bar charts
  - **Leaderboard**: Personal records on saved courses
  - **W/kg Analysis**: Power-to-weight ratio with rider classification (Cat 5 to World Tour Pro)
  - **Race**: Virtual race simulation with multiple competitors and gap analysis
  - **3D / Map**: 3D course visualization and gradient-colored route map from GPX
  - **Export**: CSV and PDF report generation

- **Dark/Light Theme Toggle**: Switch between Catppuccin dark and light color schemes

- **GPX Import**: Load real-world routes from GPX files and simulate speed on them

- **Power Zone Visualization**: Z1-Z7 colored bands on speed-vs-power chart and workout profiles (based on FTP)

- **Interval/Workout Simulator**: Define power intervals (e.g., 5min @ 300W, 2min @ 150W) and see speed, distance, and calories over time

- **Animated Ride Playback**: Watch a dot traverse the course elevation profile in real time

- **Leaderboard**: Save course results with rider/course names, track best times

- **Drag & Drop Segment Reordering**: Move segments up/down in the course profile

- **PDF Report Generation**: Multi-page PDF with simulation parameters, speed-vs-power curve, power breakdown pie chart, and course profile charts

- **Calorie Estimation**: Estimates kcal burned based on power output and metabolic efficiency (~25%)

- **Power-to-Weight Ratio Analysis**: Calculate W/kg, classify riders from Cat 5/Beginner to World Tour Pro, visualize with category bar chart and weight sensitivity curve

- **Rider Classification**: Automatic categorization based on FTP W/kg benchmarks (Cat 5 → Cat 1 → World Tour Pro)

- **Race Simulation Mode**: Add virtual competitors with custom power/weight, race them on course segments, see speed comparisons and time gaps

- **3D Course Visualization**: Matplotlib 3D terrain view with gradient-colored path, start/finish markers

- **Gradient-Colored Route Map**: Load GPX files and visualize routes colored by grade percentage with colorbar legend

- **Input Validation**: Real-time feedback for out-of-range parameter values

- **Unit Toggle**: Switch between metric (km/h, kg, m) and imperial (mph, lbs, ft)

- **Presets**: Quick-load common scenarios; save/load custom presets to JSON files

- **CSV Export**: Export results, course profiles, workouts, and scenario comparisons

## Installation

### Requirements
- Python 3.10+
- tkinter (usually included with Python; on macOS with Homebrew, install `python-tk`)
- matplotlib, numpy, gpxpy

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
2. **Simulation tab**: Adjust rider, bike, and course parameters using sliders; view real-time speed, power zone, calorie rate, power curve with zone bands, and breakdown chart
3. **Course Profile tab**: Add segments, import GPX files, click "Run Profile" to see speed and elevation plots, use "Animate" for ride playback. Reorder segments with Move Up/Down buttons
4. **Workout tab**: Define interval steps (power + duration), click "Run Workout" to see zone-colored power profile and speed chart with calorie/work summary
5. **Compare tab**: Click "Add Current as Scenario" to snapshot settings, compare multiple scenarios visually
6. **Leaderboard tab**: Save course results, track personal records
7. **Export tab**: Export results to CSV or generate a multi-page PDF report
8. Use the **Light Mode / Dark Mode** button to toggle themes
9. Use **Presets** to quickly load common riding scenarios, or save your own

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

Calories: work_kJ / metabolic_efficiency × 0.239 kcal/kJ
```

Speed is found iteratively using Newton's method to solve this nonlinear equation.

## Running Tests

```bash
python -m unittest tests -v
```

97 unit tests covering physics, validation, unit conversions, wind direction, wheel inertia, course profiles, workout simulation, power zones, GPX parsing, leaderboard, preset save/load, CSV/workout export, power-to-weight ratio, rider classification, race simulation, and GPX coordinate parsing.

## License

MIT License
