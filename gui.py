"""Bike Power Simulator — Tkinter GUI.

Provides an interactive tabbed interface for adjusting rider, bike, and course
parameters, running course profiles, comparing scenarios, and exporting results.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import Any

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from simulation import (
    BikeParams,
    CourseParams,
    CourseSegment,
    PRESETS,
    Preset,
    RiderParams,
    RidingPosition,
    SimulationResult,
    TireType,
    export_comparison_csv,
    export_course_csv,
    export_result_csv,
    load_presets,
    save_presets,
    simulate_course_profile,
    solve_speed,
    speed_vs_power_curve,
    validate_all,
    # Unit conversions
    kg_to_lbs, lbs_to_kg,
    cm_to_inches, inches_to_cm,
    m_to_feet, feet_to_m,
    celsius_to_fahrenheit, fahrenheit_to_celsius,
    kmh_to_mph, mph_to_kmh,
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
ERROR_FG = "#f38ba8"

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
    variable: tk.DoubleVar,
    command: Any = None,
) -> ttk.Scale:
    return ttk.Scale(
        parent, from_=from_, to=to, orient=tk.HORIZONTAL,
        variable=variable, command=command,
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class BikeSimApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Bike Power Simulator")
        self.configure(bg=BG)
        self.minsize(1200, 800)
        self.geometry("1400x900")

        self._imperial = False
        self._scenarios: list[tuple[str, RiderParams, BikeParams, CourseParams, SimulationResult]] = []
        self._custom_presets: list[Preset] = []
        self._course_segments: list[dict[str, tk.DoubleVar]] = []

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
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=ENTRY_BG, foreground=FG,
                        padding=[12, 4], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", BG)])
        style.configure("Result.TLabel", font=("Segoe UI", 28, "bold"),
                        foreground=ACCENT2, background=BG)
        style.configure("ResultUnit.TLabel", font=("Segoe UI", 14),
                        foreground=FG, background=BG)
        style.configure("Preset.TButton", font=("Segoe UI", 9), padding=4)
        style.configure("Small.TLabel", font=("Segoe UI", 9), foreground="#a6adc8",
                        background=BG)
        style.configure("Error.TLabel", font=("Segoe UI", 9), foreground=ERROR_FG,
                        background=BG)
        style.configure("Treeview", background=ENTRY_BG, foreground=FG,
                        fieldbackground=ENTRY_BG, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=BG_LIGHT, foreground=FG,
                        font=("Segoe UI", 9, "bold"))

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
        self.var_wind_dir = tk.DoubleVar(value=0)
        self.var_elevation = tk.DoubleVar(value=100)
        self.var_temperature = tk.DoubleVar(value=20)
        self.var_auto_update = tk.BooleanVar(value=True)
        self.var_wheel_mass = tk.DoubleVar(value=1.8)
        self.var_wheel_radius = tk.DoubleVar(value=0.34)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_main_tab()
        self._build_course_tab()
        self._build_compare_tab()
        self._build_export_tab()

    # ==================================================================
    # TAB 1: Main Simulation
    # ==================================================================
    def _build_main_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  Simulation  ")

        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(4, 4))

        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 4))

        # Error label at top of left panel
        self.lbl_error = ttk.Label(left, text="", style="Error.TLabel", wraplength=280)
        self.lbl_error.pack(fill=tk.X, pady=(0, 2))

        self._build_preset_bar(left)
        self._build_rider_section(left)
        self._build_bike_section(left)
        self._build_course_section(left)
        self._build_controls(left)
        self._build_results(right)
        self._build_charts(right)

    def _build_preset_bar(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Presets")
        frame.pack(fill=tk.X, pady=(0, 6))
        inner = ttk.Frame(frame)
        inner.pack(fill=tk.X, padx=6, pady=4)

        all_presets = PRESETS + self._custom_presets
        for i, preset in enumerate(all_presets):
            btn = ttk.Button(
                inner, text=preset.name, style="Preset.TButton",
                command=lambda p=preset: self._apply_preset(p),
            )
            btn.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

        # Save/Load preset buttons
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, padx=6, pady=(0, 4))
        ttk.Button(btn_row, text="Save Preset", style="Preset.TButton",
                   command=self._save_current_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Load Presets", style="Preset.TButton",
                   command=self._load_presets_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Export Presets", style="Preset.TButton",
                   command=self._export_presets_file).pack(side=tk.LEFT, padx=2)

    def _build_rider_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Rider")
        frame.pack(fill=tk.X, pady=(0, 6))
        self._slider_power = self._add_slider(frame, "Power (W)", self.var_power, 0, 1500, 1)
        self._slider_rw = self._add_slider(frame, "Weight (kg)", self.var_rider_weight, 30, 150, 0.5)
        self._slider_rh = self._add_slider(frame, "Height (cm)", self.var_rider_height, 140, 210, 1)

    def _build_bike_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Bike")
        frame.pack(fill=tk.X, pady=(0, 6))
        self._slider_bw = self._add_slider(frame, "Bike Weight (kg)", self.var_bike_weight, 3, 25, 0.1)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, padx=8, pady=2)
        _label(row, "Tire Type:").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_tire,
                     values=[t.value for t in TireType],
                     state="readonly", width=18).pack(side=tk.RIGHT)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, padx=8, pady=2)
        _label(row2, "Position:").pack(side=tk.LEFT)
        ttk.Combobox(row2, textvariable=self.var_position,
                     values=[p.value for p in RidingPosition],
                     state="readonly", width=18).pack(side=tk.RIGHT)

        self._slider_eff = self._add_slider(frame, "Drivetrain Eff. (%)", self.var_efficiency, 85, 100, 0.5)
        self._slider_wm = self._add_slider(frame, "Wheel Mass (kg)", self.var_wheel_mass, 0.5, 4.0, 0.1)

    def _build_course_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Course / Environment")
        frame.pack(fill=tk.X, pady=(0, 6))
        self._slider_grade = self._add_slider(frame, "Grade (%)", self.var_grade, -20, 25, 0.1)
        self._slider_wind = self._add_slider(frame, "Wind Speed (km/h)", self.var_wind, 0, 80, 1)
        self._slider_wind_dir = self._add_slider(frame, "Wind Dir (0=head 180=tail)", self.var_wind_dir, 0, 360, 5)
        self._slider_elev = self._add_slider(frame, "Elevation (m)", self.var_elevation, 0, 5000, 10)
        self._slider_temp = self._add_slider(frame, "Temperature (\u00b0C)", self.var_temperature, -10, 50, 1)

    def _build_controls(self, parent: tk.Widget) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Checkbutton(frame, text="Auto-update",
                        variable=self.var_auto_update).pack(side=tk.LEFT, padx=4)
        ttk.Button(frame, text="Calculate",
                   command=self._run_simulation).pack(side=tk.RIGHT, padx=4)

        frame2 = ttk.Frame(parent)
        frame2.pack(fill=tk.X, pady=(4, 0))
        self._unit_btn = ttk.Button(frame2, text="Switch to Imperial",
                                     command=self._toggle_units)
        self._unit_btn.pack(side=tk.LEFT, padx=4)

    def _build_results(self, parent: tk.Widget) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 8))

        self.lbl_speed_primary = ttk.Label(frame, text="0.0", style="Result.TLabel")
        self.lbl_speed_primary.pack(side=tk.LEFT, padx=(8, 0))
        self.lbl_speed_unit1 = ttk.Label(frame, text="km/h", style="ResultUnit.TLabel")
        self.lbl_speed_unit1.pack(side=tk.LEFT, anchor=tk.S, pady=(0, 6))

        ttk.Label(frame, text="  |  ", style="ResultUnit.TLabel").pack(
            side=tk.LEFT, anchor=tk.S, pady=(0, 6))

        self.lbl_speed_secondary = ttk.Label(frame, text="0.0", style="Result.TLabel")
        self.lbl_speed_secondary.pack(side=tk.LEFT)
        self.lbl_speed_unit2 = ttk.Label(frame, text="mph", style="ResultUnit.TLabel")
        self.lbl_speed_unit2.pack(side=tk.LEFT, anchor=tk.S, pady=(0, 6))

        detail_frame = ttk.Frame(parent)
        detail_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.lbl_details = ttk.Label(detail_frame, text="", style="Small.TLabel",
                                     wraplength=600, justify=tk.LEFT)
        self.lbl_details.pack(anchor=tk.W)

    def _build_charts(self, parent: tk.Widget) -> None:
        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor=BG)
        self.fig.subplots_adjust(hspace=0.45, left=0.10, right=0.95,
                                 top=0.94, bottom=0.10)
        self.ax_curve = self.fig.add_subplot(211)
        self.ax_pie = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ==================================================================
    # TAB 2: Course Profile
    # ==================================================================
    def _build_course_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  Course Profile  ")

        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=4)

        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Segment list
        seg_frame = ttk.LabelFrame(left, text="Course Segments")
        seg_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        btn_row = ttk.Frame(seg_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_row, text="Add Segment", style="Preset.TButton",
                   command=self._add_course_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Remove Last", style="Preset.TButton",
                   command=self._remove_course_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Run Profile", command=self._run_course_profile).pack(
            side=tk.RIGHT, padx=2)

        # Scrollable segment entries
        seg_canvas = tk.Canvas(seg_frame, bg=BG, highlightthickness=0, width=320)
        seg_scrollbar = ttk.Scrollbar(seg_frame, orient=tk.VERTICAL, command=seg_canvas.yview)
        self._seg_inner = ttk.Frame(seg_canvas)
        self._seg_inner.bind("<Configure>",
                             lambda e: seg_canvas.configure(scrollregion=seg_canvas.bbox("all")))
        seg_canvas.create_window((0, 0), window=self._seg_inner, anchor="nw")
        seg_canvas.configure(yscrollcommand=seg_scrollbar.set)
        seg_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        seg_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add 3 default segments
        for grade in [0, 5, -3]:
            self._add_course_segment(grade=grade)

        # Course results
        self.lbl_course_summary = ttk.Label(left, text="", style="Small.TLabel",
                                             wraplength=300, justify=tk.LEFT)
        self.lbl_course_summary.pack(fill=tk.X, pady=4)

        # Course charts
        self.fig_course = Figure(figsize=(7, 5), dpi=100, facecolor=BG)
        self.fig_course.subplots_adjust(hspace=0.40, left=0.10, right=0.95,
                                         top=0.94, bottom=0.10)
        self.ax_course_speed = self.fig_course.add_subplot(211)
        self.ax_course_elev = self.fig_course.add_subplot(212)
        self.canvas_course = FigureCanvasTkAgg(self.fig_course, master=right)
        self.canvas_course.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_course_segment(self, grade: float = 0.0) -> None:
        idx = len(self._course_segments)
        seg_vars: dict[str, tk.DoubleVar] = {
            "distance": tk.DoubleVar(value=1000),
            "grade": tk.DoubleVar(value=grade),
            "wind": tk.DoubleVar(value=0),
            "wind_dir": tk.DoubleVar(value=0),
            "elevation": tk.DoubleVar(value=100),
            "temperature": tk.DoubleVar(value=20),
        }
        self._course_segments.append(seg_vars)

        frame = ttk.LabelFrame(self._seg_inner, text=f"Segment {idx + 1}")
        frame.pack(fill=tk.X, padx=4, pady=2)

        entries = [
            ("Dist (m):", seg_vars["distance"]),
            ("Grade (%):", seg_vars["grade"]),
            ("Wind (km/h):", seg_vars["wind"]),
            ("Wind Dir:", seg_vars["wind_dir"]),
            ("Elev (m):", seg_vars["elevation"]),
            ("Temp (\u00b0C):", seg_vars["temperature"]),
        ]
        for lbl_text, var in entries:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, padx=4, pady=1)
            _label(row, lbl_text).pack(side=tk.LEFT)
            entry = ttk.Entry(row, textvariable=var, width=8)
            entry.pack(side=tk.RIGHT)

    def _remove_course_segment(self) -> None:
        if self._course_segments:
            self._course_segments.pop()
            children = self._seg_inner.winfo_children()
            if children:
                children[-1].destroy()

    def _run_course_profile(self) -> None:
        rider, bike, _ = self._gather_params()
        segments: list[CourseSegment] = []
        for sv in self._course_segments:
            try:
                segments.append(CourseSegment(
                    distance_m=sv["distance"].get(),
                    grade_pct=sv["grade"].get(),
                    headwind_kmh=sv["wind"].get(),
                    wind_direction_deg=sv["wind_dir"].get(),
                    elevation_m=sv["elevation"].get(),
                    temperature_c=sv["temperature"].get(),
                ))
            except (tk.TclError, ValueError):
                messagebox.showerror("Input Error", f"Invalid values in segment")
                return

        if not segments:
            messagebox.showwarning("No Segments", "Add at least one course segment.")
            return

        result = simulate_course_profile(rider, bike, segments)
        self._last_course_result = result

        mins = int(result.total_time_s // 60)
        secs = int(result.total_time_s % 60)
        summary = (
            f"Total: {result.total_distance_m / 1000:.2f} km  |  "
            f"Time: {mins}:{secs:02d}\n"
            f"Avg Speed: {result.avg_speed_kmh:.1f} km/h\n"
            f"Elev Gain: {result.total_elevation_gain_m:.0f} m  |  "
            f"Loss: {result.total_elevation_loss_m:.0f} m"
        )
        self.lbl_course_summary.configure(text=summary)
        self._update_course_charts(result)

    def _update_course_charts(self, result: Any) -> None:
        distances_km = [d / 1000 for d in result.distances_cumulative]

        ax1 = self.ax_course_speed
        ax1.clear()
        ax1.set_facecolor(BG_LIGHT)
        mid_distances = [(distances_km[i] + distances_km[i + 1]) / 2
                         for i in range(len(result.speeds))]
        ax1.bar(mid_distances, result.speeds, width=[
            distances_km[i + 1] - distances_km[i] for i in range(len(result.speeds))
        ], color=ACCENT, alpha=0.8, edgecolor=BORDER)
        ax1.axhline(result.avg_speed_kmh, color=ACCENT4, linestyle="--", linewidth=1,
                    label=f"Avg: {result.avg_speed_kmh:.1f} km/h")
        ax1.set_xlabel("Distance (km)", color=FG, fontsize=9)
        ax1.set_ylabel("Speed (km/h)", color=FG, fontsize=9)
        ax1.set_title("Speed by Segment", color=FG, fontsize=11, fontweight="bold")
        ax1.tick_params(colors=FG, labelsize=8)
        ax1.legend(fontsize=8, facecolor=BG_LIGHT, edgecolor=BORDER, labelcolor=FG)
        ax1.grid(True, alpha=0.2, color=FG)
        for spine in ax1.spines.values():
            spine.set_color(BORDER)

        ax2 = self.ax_course_elev
        ax2.clear()
        ax2.set_facecolor(BG_LIGHT)
        ax2.fill_between(distances_km, result.elevations, alpha=0.3, color=ACCENT2)
        ax2.plot(distances_km, result.elevations, color=ACCENT2, linewidth=2)
        ax2.set_xlabel("Distance (km)", color=FG, fontsize=9)
        ax2.set_ylabel("Elevation (m)", color=FG, fontsize=9)
        ax2.set_title("Elevation Profile", color=FG, fontsize=11, fontweight="bold")
        ax2.tick_params(colors=FG, labelsize=8)
        ax2.grid(True, alpha=0.2, color=FG)
        for spine in ax2.spines.values():
            spine.set_color(BORDER)

        self.canvas_course.draw_idle()

    # ==================================================================
    # TAB 3: Compare Scenarios
    # ==================================================================
    def _build_compare_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  Compare  ")

        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=8, pady=4)

        ttk.Button(top, text="Add Current as Scenario",
                   command=self._add_scenario).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Clear All",
                   command=self._clear_scenarios).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Export Comparison CSV",
                   command=self._export_comparison).pack(side=tk.RIGHT, padx=4)

        # Treeview for scenarios
        cols = ("Name", "Power", "Weight", "Grade", "Wind", "Speed (km/h)", "Speed (mph)")
        self._compare_tree = ttk.Treeview(tab, columns=cols, show="headings", height=8)
        for col in cols:
            self._compare_tree.heading(col, text=col)
            self._compare_tree.column(col, width=100, anchor=tk.CENTER)
        self._compare_tree.pack(fill=tk.X, padx=8, pady=4)

        # Comparison chart
        self.fig_compare = Figure(figsize=(7, 3), dpi=100, facecolor=BG)
        self.fig_compare.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.20)
        self.ax_compare = self.fig_compare.add_subplot(111)
        self.canvas_compare = FigureCanvasTkAgg(self.fig_compare, master=tab)
        self.canvas_compare.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8)

    def _add_scenario(self) -> None:
        rider, bike, course = self._gather_params()
        result = solve_speed(rider, bike, course)
        name = f"Scenario {len(self._scenarios) + 1}"
        self._scenarios.append((name, rider, bike, course, result))

        self._compare_tree.insert("", tk.END, values=(
            name,
            f"{rider.power_watts:.0f} W",
            f"{rider.weight_kg + bike.weight_kg:.1f} kg",
            f"{course.grade_pct:.1f}%",
            f"{course.headwind_kmh:.0f} km/h",
            f"{result.speed_kmh:.1f}",
            f"{result.speed_mph:.1f}",
        ))
        self._update_compare_chart()

    def _clear_scenarios(self) -> None:
        self._scenarios.clear()
        for item in self._compare_tree.get_children():
            self._compare_tree.delete(item)
        self.ax_compare.clear()
        self.canvas_compare.draw_idle()

    def _update_compare_chart(self) -> None:
        ax = self.ax_compare
        ax.clear()
        ax.set_facecolor(BG_LIGHT)

        if not self._scenarios:
            self.canvas_compare.draw_idle()
            return

        names = [s[0] for s in self._scenarios]
        speeds = [s[4].speed_kmh for s in self._scenarios]
        colors = [CHART_COLOURS[i % len(CHART_COLOURS)] for i in range(len(names))]

        bars = ax.bar(names, speeds, color=colors, edgecolor=BORDER, alpha=0.85)
        for bar, spd in zip(bars, speeds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{spd:.1f}", ha="center", va="bottom", color=FG, fontsize=8)

        ax.set_ylabel("Speed (km/h)", color=FG, fontsize=9)
        ax.set_title("Scenario Comparison", color=FG, fontsize=11, fontweight="bold")
        ax.tick_params(colors=FG, labelsize=8)
        ax.grid(True, alpha=0.2, color=FG, axis="y")
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        self.canvas_compare.draw_idle()

    def _export_comparison(self) -> None:
        if not self._scenarios:
            messagebox.showwarning("No Data", "Add scenarios before exporting.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            export_comparison_csv(self._scenarios, path)
            messagebox.showinfo("Exported", f"Comparison saved to {path}")

    # ==================================================================
    # TAB 4: Export
    # ==================================================================
    def _build_export_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  Export  ")

        info = ttk.Frame(tab)
        info.pack(fill=tk.X, padx=20, pady=20)

        _label(info, "Export current simulation or course profile results.",
               font=("Segoe UI", 12)).pack(anchor=tk.W, pady=4)

        ttk.Button(info, text="Export Current Result to CSV",
                   command=self._export_result_csv).pack(anchor=tk.W, pady=4)
        ttk.Button(info, text="Export Course Profile to CSV",
                   command=self._export_course_csv).pack(anchor=tk.W, pady=4)
        ttk.Button(info, text="Export Comparison to CSV",
                   command=self._export_comparison).pack(anchor=tk.W, pady=4)

    def _export_result_csv(self) -> None:
        rider, bike, course = self._gather_params()
        result = solve_speed(rider, bike, course)
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            export_result_csv(result, rider, bike, course, path)
            messagebox.showinfo("Exported", f"Results saved to {path}")

    def _export_course_csv(self) -> None:
        if not hasattr(self, "_last_course_result"):
            messagebox.showwarning("No Data", "Run a course profile first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            export_course_csv(self._last_course_result, path)
            messagebox.showinfo("Exported", f"Course profile saved to {path}")

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
    ) -> dict[str, Any]:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=2)

        lbl_widget = _label(row, f"{label}:")
        lbl_widget.pack(side=tk.LEFT)
        val_label = _label(row, f"{var.get():.1f}")
        val_label.pack(side=tk.RIGHT, padx=(4, 0))

        scale = _make_scale(
            row, from_, to, var,
            command=lambda v, vl=val_label, vr=var, r=resolution: self._on_scale(v, vl, vr, r),
        )
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=4)

        def _on_trace(*_args: object, vl: ttk.Label = val_label,
                       vr: tk.DoubleVar = var, r: float = resolution) -> None:
            v = vr.get()
            if r >= 1:
                vl.configure(text=f"{v:.0f}")
            else:
                decimals = max(1, len(str(r).split(".")[-1]))
                vl.configure(text=f"{v:.{decimals}f}")
        var.trace_add("write", _on_trace)

        return {"label_widget": lbl_widget, "val_label": val_label,
                "scale": scale, "from": from_, "to": to, "resolution": resolution}

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
    # Unit toggle
    # ------------------------------------------------------------------
    def _toggle_units(self) -> None:
        self._imperial = not self._imperial
        if self._imperial:
            self._unit_btn.configure(text="Switch to Metric")
            self.lbl_speed_unit1.configure(text="mph")
            self.lbl_speed_unit2.configure(text="km/h")
        else:
            self._unit_btn.configure(text="Switch to Imperial")
            self.lbl_speed_unit1.configure(text="km/h")
            self.lbl_speed_unit2.configure(text="mph")
        self._run_simulation()

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def _apply_preset(self, preset: Preset) -> None:
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
        self.var_wind_dir.set(c.wind_direction_deg)
        self.var_elevation.set(c.elevation_m)
        self.var_temperature.set(c.temperature_c)
        self.var_wheel_mass.set(b.wheel_mass_kg)
        self._run_simulation()

    def _save_current_preset(self) -> None:
        name = simpledialog.askstring("Preset Name", "Enter a name for this preset:",
                                      parent=self)
        if not name:
            return
        rider, bike, course = self._gather_params()
        preset = Preset(name=name, rider=rider, bike=bike, course=course)
        self._custom_presets.append(preset)
        messagebox.showinfo("Saved", f"Preset '{name}' saved. Use Export Presets to save to file.")

    def _load_presets_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            try:
                loaded = load_presets(path)
                self._custom_presets.extend(loaded)
                messagebox.showinfo("Loaded", f"Loaded {len(loaded)} presets from file.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load presets: {e}")

    def _export_presets_file(self) -> None:
        all_presets = PRESETS + self._custom_presets
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            save_presets(all_presets, path)
            messagebox.showinfo("Exported", f"Presets saved to {path}")

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
            wheel_mass_kg=self.var_wheel_mass.get(),
        )
        course = CourseParams(
            grade_pct=self.var_grade.get(),
            headwind_kmh=self.var_wind.get(),
            wind_direction_deg=self.var_wind_dir.get(),
            elevation_m=self.var_elevation.get(),
            temperature_c=self.var_temperature.get(),
        )
        return rider, bike, course

    def _run_simulation(self) -> None:
        rider, bike, course = self._gather_params()

        errors = validate_all(rider, bike, course)
        if errors:
            self.lbl_error.configure(text="\n".join(errors))
        else:
            self.lbl_error.configure(text="")

        result = solve_speed(rider, bike, course)

        if self._imperial:
            self.lbl_speed_primary.configure(text=f"{result.speed_mph:.1f}")
            self.lbl_speed_secondary.configure(text=f"{result.speed_kmh:.1f}")
        else:
            self.lbl_speed_primary.configure(text=f"{result.speed_kmh:.1f}")
            self.lbl_speed_secondary.configure(text=f"{result.speed_mph:.1f}")

        eff_wind = result.forces.aero  # just for display
        details = (
            f"Air density: {result.air_density:.3f} kg/m\u00b3  |  "
            f"CdA: {result.cda:.4f} m\u00b2  |  Crr: {result.crr:.4f}\n"
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
        ax = self.ax_curve
        ax.clear()
        ax.set_facecolor(BG_LIGHT)

        powers, speeds = speed_vs_power_curve(rider, bike, course)
        speed_label = "Speed (km/h)" if not self._imperial else "Speed (mph)"
        if self._imperial:
            speeds = speeds * 0.621371

        ax.plot(powers, speeds, color=ACCENT, linewidth=2, label="Speed")
        current_speed = result.speed_mph if self._imperial else result.speed_kmh
        ax.axvline(rider.power_watts, color=ACCENT4, linestyle="--", linewidth=1,
                   alpha=0.7, label=f"{rider.power_watts:.0f} W")
        ax.axhline(current_speed, color=ACCENT2, linestyle=":", linewidth=1, alpha=0.5)
        ax.scatter([rider.power_watts], [current_speed], color=ACCENT2, s=60, zorder=5)

        ax.set_xlabel("Power (W)", color=FG, fontsize=9)
        ax.set_ylabel(speed_label, color=FG, fontsize=9)
        ax.set_title("Speed vs Power", color=FG, fontsize=11, fontweight="bold")
        ax.tick_params(colors=FG, labelsize=8)
        ax.legend(fontsize=8, loc="lower right",
                  facecolor=BG_LIGHT, edgecolor=BORDER, labelcolor=FG)
        ax.grid(True, alpha=0.2, color=FG)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

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
                sizes, labels=labels, colors=colors, autopct="%1.0f%%",
                startangle=90, textprops={"color": FG, "fontsize": 8},
                pctdistance=0.75,
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_color(BG)
                at.set_fontweight("bold")
        ax2.set_title("Power Breakdown", color=FG, fontsize=11, fontweight="bold")

        self.canvas.draw_idle()
