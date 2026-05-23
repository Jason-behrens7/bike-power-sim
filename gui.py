"""Bike Power Simulator — Tkinter GUI.

Provides an interactive interface for adjusting rider, bike, and course
parameters and viewing real-time speed results with charts.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from simulation import (
    BikeParams,
    CourseParams,
    PRESETS,
    RiderParams,
    RidingPosition,
    SimulationResult,
    TireType,
    solve_speed,
    speed_vs_power_curve,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG = "#1e1e2e"
BG_LIGHT = "#2a2a3c"
FG = "#cdd6f4"
ACCENT = "#89b4fa"
ACCENT2 = "#a6e3a1"
ACCENT3 = "#f9e2af"
ACCENT4 = "#f38ba8"
BORDER = "#45475a"
ENTRY_BG = "#313244"

CHART_COLOURS = [ACCENT, ACCENT2, ACCENT3, ACCENT4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label(parent: tk.Widget, text: str, **kw: Any) -> ttk.Label:
    return ttk.Label(parent, text=text, **kw)


def _make_scale(
    parent: tk.Widget,
    from_: float,
    to: float,
    resolution: float,
    variable: tk.DoubleVar,
    command: Any = None,
) -> ttk.Scale:
    s = ttk.Scale(
        parent,
        from_=from_,
        to=to,
        orient=tk.HORIZONTAL,
        variable=variable,
        command=command,
    )
    return s


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class BikeSimApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Bike Power Simulator")
        self.configure(bg=BG)
        self.minsize(1100, 750)
        self.geometry("1280x800")

        self._setup_style()
        self._create_variables()
        self._build_ui()
        self._run_simulation()

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------
    def _setup_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=BG, foreground=FG, fieldbackground=ENTRY_BG,
                        borderwidth=0, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TLabelframe", background=BG, foreground=ACCENT,
                        borderwidth=1, relief="groove")
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                        font=("Segoe UI", 11, "bold"))
        style.configure("TScale", background=BG, troughcolor=ENTRY_BG)
        style.configure("TButton", background=ACCENT, foreground=BG,
                        font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton",
                  background=[("active", ACCENT2)],
                  foreground=[("active", BG)])
        style.configure("TCombobox", fieldbackground=ENTRY_BG, background=ENTRY_BG,
                        foreground=FG, arrowcolor=FG)
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.configure("Result.TLabel", font=("Segoe UI", 28, "bold"),
                        foreground=ACCENT2, background=BG)
        style.configure("ResultUnit.TLabel", font=("Segoe UI", 14),
                        foreground=FG, background=BG)
        style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"),
                        foreground=ACCENT, background=BG)
        style.configure("Preset.TButton", font=("Segoe UI", 9), padding=4)
        style.configure("Small.TLabel", font=("Segoe UI", 9), foreground="#a6adc8",
                        background=BG)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    def _create_variables(self) -> None:
        self.var_power = tk.DoubleVar(value=200)
        self.var_rider_weight = tk.DoubleVar(value=75)
        self.var_rider_height = tk.DoubleVar(value=178)
        self.var_bike_weight = tk.DoubleVar(value=8)
        self.var_tire = tk.StringVar(value=TireType.ROAD_TRAINING.value)
        self.var_position = tk.StringVar(value=RidingPosition.HOODS.value)
        self.var_efficiency = tk.DoubleVar(value=97)
        self.var_grade = tk.DoubleVar(value=0)
        self.var_wind = tk.DoubleVar(value=0)
        self.var_elevation = tk.DoubleVar(value=100)
        self.var_temperature = tk.DoubleVar(value=20)
        self.var_auto_update = tk.BooleanVar(value=True)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Top-level paned layout
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 4))

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self._build_preset_bar(left)
        self._build_rider_section(left)
        self._build_bike_section(left)
        self._build_course_section(left)
        self._build_controls(left)
        self._build_results(right)
        self._build_charts(right)

    # -- presets ---
    def _build_preset_bar(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Presets")
        frame.pack(fill=tk.X, pady=(0, 6))
        inner = ttk.Frame(frame)
        inner.pack(fill=tk.X, padx=6, pady=4)
        for i, preset in enumerate(PRESETS):
            btn = ttk.Button(
                inner,
                text=preset.name,
                style="Preset.TButton",
                command=lambda p=preset: self._apply_preset(p),
            )
            btn.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

    # -- rider ---
    def _build_rider_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Rider")
        frame.pack(fill=tk.X, pady=(0, 6))

        self._add_slider(frame, "Power (W)", self.var_power, 0, 1500, 1)
        self._add_slider(frame, "Weight (kg)", self.var_rider_weight, 30, 150, 0.5)
        self._add_slider(frame, "Height (cm)", self.var_rider_height, 140, 210, 1)

    # -- bike ---
    def _build_bike_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Bike")
        frame.pack(fill=tk.X, pady=(0, 6))

        self._add_slider(frame, "Bike Weight (kg)", self.var_bike_weight, 3, 25, 0.1)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, padx=8, pady=2)
        _label(row, "Tire Type:").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.var_tire,
            values=[t.value for t in TireType],
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, padx=8, pady=2)
        _label(row2, "Position:").pack(side=tk.LEFT)
        ttk.Combobox(
            row2,
            textvariable=self.var_position,
            values=[p.value for p in RidingPosition],
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT)

        self._add_slider(frame, "Drivetrain Eff. (%)", self.var_efficiency, 85, 100, 0.5)

    # -- course ---
    def _build_course_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Course / Environment")
        frame.pack(fill=tk.X, pady=(0, 6))

        self._add_slider(frame, "Grade (%)", self.var_grade, -20, 25, 0.1)
        self._add_slider(frame, "Headwind (km/h)", self.var_wind, -50, 50, 1)
        self._add_slider(frame, "Elevation (m)", self.var_elevation, 0, 5000, 10)
        self._add_slider(frame, "Temperature (°C)", self.var_temperature, -10, 50, 1)

    # -- controls ---
    def _build_controls(self, parent: tk.Widget) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(4, 0))

        ttk.Checkbutton(
            frame,
            text="Auto-update",
            variable=self.var_auto_update,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(frame, text="Calculate", command=self._run_simulation).pack(
            side=tk.RIGHT, padx=4
        )

    # -- results ---
    def _build_results(self, parent: tk.Widget) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 8))

        self.lbl_speed_kmh = ttk.Label(frame, text="0.0", style="Result.TLabel")
        self.lbl_speed_kmh.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(frame, text="km/h", style="ResultUnit.TLabel").pack(
            side=tk.LEFT, anchor=tk.S, pady=(0, 6)
        )

        ttk.Label(frame, text="  |  ", style="ResultUnit.TLabel").pack(
            side=tk.LEFT, anchor=tk.S, pady=(0, 6)
        )

        self.lbl_speed_mph = ttk.Label(frame, text="0.0", style="Result.TLabel")
        self.lbl_speed_mph.pack(side=tk.LEFT)
        ttk.Label(frame, text="mph", style="ResultUnit.TLabel").pack(
            side=tk.LEFT, anchor=tk.S, pady=(0, 6)
        )

        # Detail labels
        detail_frame = ttk.Frame(parent)
        detail_frame.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.lbl_details = ttk.Label(detail_frame, text="", style="Small.TLabel",
                                     wraplength=600, justify=tk.LEFT)
        self.lbl_details.pack(anchor=tk.W)

    # -- charts ---
    def _build_charts(self, parent: tk.Widget) -> None:
        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor=BG)
        self.fig.subplots_adjust(hspace=0.45, left=0.10, right=0.95,
                                 top=0.94, bottom=0.10)

        self.ax_curve = self.fig.add_subplot(211)
        self.ax_pie = self.fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Slider helper
    # ------------------------------------------------------------------
    def _add_slider(
        self,
        parent: tk.Widget,
        label: str,
        var: tk.DoubleVar,
        from_: float,
        to: float,
        resolution: float,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=2)

        _label(row, f"{label}:").pack(side=tk.LEFT)
        val_label = _label(row, f"{var.get():.1f}")
        val_label.pack(side=tk.RIGHT, padx=(4, 0))

        scale = _make_scale(
            row, from_, to, resolution, var,
            command=lambda v, vl=val_label, vr=var, r=resolution: self._on_scale(v, vl, vr, r),
        )
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=4)

    def _on_scale(
        self,
        value: str,
        val_label: ttk.Label,
        var: tk.DoubleVar,
        resolution: float,
    ) -> None:
        v = float(value)
        if resolution >= 1:
            v = round(v)
            var.set(v)
            val_label.configure(text=f"{v:.0f}")
        else:
            decimals = max(1, len(str(resolution).split(".")[-1]))
            v = round(v / resolution) * resolution
            var.set(v)
            val_label.configure(text=f"{v:.{decimals}f}")

        if self.var_auto_update.get():
            self.after(50, self._run_simulation)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def _apply_preset(self, preset: Any) -> None:
        r, b, c = preset.rider, preset.bike, preset.course
        self.var_power.set(r.power_watts)
        self.var_rider_weight.set(r.weight_kg)
        self.var_rider_height.set(r.height_cm)
        self.var_bike_weight.set(b.weight_kg)
        self.var_tire.set(b.tire_type.value)
        self.var_position.set(b.position.value)
        self.var_efficiency.set(b.drivetrain_efficiency * 100)
        self.var_grade.set(c.grade_pct)
        self.var_wind.set(c.headwind_kmh)
        self.var_elevation.set(c.elevation_m)
        self.var_temperature.set(c.temperature_c)
        self._run_simulation()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def _gather_params(self) -> tuple[RiderParams, BikeParams, CourseParams]:
        tire = next(t for t in TireType if t.value == self.var_tire.get())
        pos = next(p for p in RidingPosition if p.value == self.var_position.get())

        rider = RiderParams(
            power_watts=self.var_power.get(),
            weight_kg=self.var_rider_weight.get(),
            height_cm=self.var_rider_height.get(),
        )
        bike = BikeParams(
            weight_kg=self.var_bike_weight.get(),
            tire_type=tire,
            position=pos,
            drivetrain_efficiency=self.var_efficiency.get() / 100.0,
        )
        course = CourseParams(
            grade_pct=self.var_grade.get(),
            headwind_kmh=self.var_wind.get(),
            elevation_m=self.var_elevation.get(),
            temperature_c=self.var_temperature.get(),
        )
        return rider, bike, course

    def _run_simulation(self) -> None:
        rider, bike, course = self._gather_params()
        result = solve_speed(rider, bike, course)

        self.lbl_speed_kmh.configure(text=f"{result.speed_kmh:.1f}")
        self.lbl_speed_mph.configure(text=f"{result.speed_mph:.1f}")

        details = (
            f"Air density: {result.air_density:.3f} kg/m³  |  "
            f"CdA: {result.cda:.4f} m²  |  Crr: {result.crr:.4f}\n"
            f"Aero: {result.power_aero:.1f} W  |  "
            f"Rolling: {result.power_rolling:.1f} W  |  "
            f"Gravity: {result.power_gravity:.1f} W  |  "
            f"Drivetrain loss: {result.power_drivetrain_loss:.1f} W"
        )
        self.lbl_details.configure(text=details)

        self._update_charts(rider, bike, course, result)

    def _update_charts(
        self,
        rider: RiderParams,
        bike: BikeParams,
        course: CourseParams,
        result: SimulationResult,
    ) -> None:
        # --- Speed vs Power curve ---
        ax = self.ax_curve
        ax.clear()
        ax.set_facecolor(BG_LIGHT)

        powers, speeds = speed_vs_power_curve(rider, bike, course)
        ax.plot(powers, speeds, color=ACCENT, linewidth=2, label="Speed")
        ax.axvline(rider.power_watts, color=ACCENT4, linestyle="--", linewidth=1,
                   alpha=0.7, label=f"{rider.power_watts:.0f} W")
        ax.axhline(result.speed_kmh, color=ACCENT2, linestyle=":", linewidth=1,
                   alpha=0.5)
        ax.scatter([rider.power_watts], [result.speed_kmh], color=ACCENT2,
                   s=60, zorder=5)

        ax.set_xlabel("Power (W)", color=FG, fontsize=9)
        ax.set_ylabel("Speed (km/h)", color=FG, fontsize=9)
        ax.set_title("Speed vs Power", color=FG, fontsize=11, fontweight="bold")
        ax.tick_params(colors=FG, labelsize=8)
        ax.legend(fontsize=8, loc="lower right",
                  facecolor=BG_LIGHT, edgecolor=BORDER, labelcolor=FG)
        ax.grid(True, alpha=0.2, color=FG)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        # --- Power breakdown pie ---
        ax2 = self.ax_pie
        ax2.clear()
        ax2.set_facecolor(BG)

        labels = []
        sizes = []
        colors = []
        breakdown = [
            ("Aero", abs(result.power_aero), ACCENT),
            ("Rolling", abs(result.power_rolling), ACCENT2),
            ("Gravity", abs(result.power_gravity), ACCENT3),
            ("Drivetrain", abs(result.power_drivetrain_loss), ACCENT4),
        ]
        for lbl, val, col in breakdown:
            if val > 0.1:
                labels.append(f"{lbl}\n{val:.0f} W")
                sizes.append(val)
                colors.append(col)

        if sizes:
            wedges, texts, autotexts = ax2.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.0f%%",
                startangle=90,
                textprops={"color": FG, "fontsize": 8},
                pctdistance=0.75,
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_color(BG)
                at.set_fontweight("bold")
        ax2.set_title("Power Breakdown", color=FG, fontsize=11, fontweight="bold")

        self.canvas.draw_idle()
