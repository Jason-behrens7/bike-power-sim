"""Bike Power Simulator — Tkinter GUI.

Provides an interactive tabbed interface with simulation, course profiles,
workout intervals, GPX import, scenario comparison, and export.
"""

from __future__ import annotations

import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import Any

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from simulation import (
    BikeParams,
    CourseParams,
    CourseSegment,
    CriticalPowerModel,
    IntervalStep,
    PRESETS,
    Preset,
    RaceCompetitor,
    RIDER_CATEGORIES,
    RiderParams,
    RidingPosition,
    SimulationResult,
    TireType,
    ZONE_COLORS,
    ZONE_NAMES,
    classify_rider,
    estimate_calories,
    estimate_cp_from_ftp,
    export_comparison_csv,
    export_course_csv,
    export_result_csv,
    export_workout_csv,
    fit_cp_model,
    load_presets,
    optimize_pacing,
    parse_gpx,
    parse_gpx_with_coords,
    power_to_weight_ratio,
    power_zones,
    save_presets,
    simulate_course_profile,
    simulate_race,
    simulate_workout,
    solve_speed,
    speed_vs_power_curve,
    validate_all,
    w_per_kg_analysis,
    zone_for_power,
    # Unit conversions
    kg_to_lbs, lbs_to_kg,
    cm_to_inches, inches_to_cm,
    m_to_feet, feet_to_m,
    celsius_to_fahrenheit, fahrenheit_to_celsius,
    kmh_to_mph, mph_to_kmh,
)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
# Dark theme (Catppuccin Mocha)
DARK = {
    "BG": "#1e1e2e", "BG_LIGHT": "#262637", "FG": "#cdd6f4",
    "FG_DIM": "#6c7086", "ACCENT": "#89b4fa", "ACCENT2": "#a6e3a1",
    "ACCENT3": "#f9e2af", "ACCENT4": "#f38ba8", "BORDER": "#393950",
    "ENTRY_BG": "#313244", "ERROR_FG": "#f38ba8",
    "SURFACE0": "#313244", "SURFACE1": "#45475a",
}
# Light theme (Catppuccin Latte)
LIGHT = {
    "BG": "#eff1f5", "BG_LIGHT": "#e6e9ef", "FG": "#4c4f69",
    "FG_DIM": "#8c8fa1", "ACCENT": "#1e66f5", "ACCENT2": "#40a02b",
    "ACCENT3": "#df8e1d", "ACCENT4": "#d20f39", "BORDER": "#ccd0da",
    "ENTRY_BG": "#ccd0da", "ERROR_FG": "#d20f39",
    "SURFACE0": "#ccd0da", "SURFACE1": "#bcc0cc",
}

CHART_COLOURS_KEYS = ["ACCENT", "ACCENT2", "ACCENT3", "ACCENT4"]




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


def _style_ax(ax: Any, theme: dict, bg_key: str = "BG_LIGHT") -> None:
    """Apply theme colours to a matplotlib axes."""
    ax.set_facecolor(theme[bg_key])
    ax.tick_params(colors=theme["FG"], labelsize=8)
    ax.grid(True, alpha=0.2, color=theme["FG"])
    for spine in ax.spines.values():
        spine.set_color(theme["BORDER"])


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class BikeSimApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Bike Power Simulator")
        self.minsize(1200, 800)
        self.geometry("1400x900")

        self._theme = DARK
        self._imperial = False
        self._scenarios: list[tuple[str, RiderParams, BikeParams, CourseParams, SimulationResult]] = []
        self._custom_presets: list[Preset] = []
        self._course_segments: list[dict[str, tk.DoubleVar]] = []
        self._interval_steps: list[dict[str, tk.DoubleVar]] = []
        self._animation_id: str | None = None
        self._last_course_result: Any = None

        self._apply_theme()
        self._create_variables()
        self._build_ui()
        self._run_simulation()

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------
    def _apply_theme(self) -> None:
        t = self._theme
        self.configure(bg=t["BG"])

        style = ttk.Style(self)
        style.theme_use("clam")

        _font = ("Helvetica", 10)
        _font_bold = ("Helvetica", 10, "bold")
        _font_sm = ("Helvetica", 9)
        _font_heading = ("Helvetica", 11, "bold")

        style.configure(".", background=t["BG"], foreground=t["FG"],
                        fieldbackground=t["ENTRY_BG"], borderwidth=0,
                        font=_font)
        style.configure("TFrame", background=t["BG"])
        style.configure("TLabel", background=t["BG"], foreground=t["FG"])
        style.configure("TLabelframe", background=t["BG"], foreground=t["ACCENT"],
                        borderwidth=1, relief="flat")
        style.configure("TLabelframe.Label", background=t["BG"], foreground=t["ACCENT"],
                        font=_font_heading)
        style.configure("TScale", background=t["BG"],
                        troughcolor=t["SURFACE0"])
        style.configure("TButton", background=t["ACCENT"], foreground="#ffffff",
                        font=_font_bold, padding=(10, 5))
        style.map("TButton",
                  background=[("active", t["ACCENT2"]), ("disabled", t["BORDER"])],
                  foreground=[("active", "#ffffff"), ("disabled", t.get("FG_DIM", t["BORDER"]))])
        style.configure("Secondary.TButton", background=t["SURFACE0"],
                        foreground=t["FG"], font=_font, padding=(8, 4))
        style.map("Secondary.TButton",
                  background=[("active", t["SURFACE1"])])
        style.configure("TCombobox", fieldbackground=t["ENTRY_BG"],
                        background=t["ENTRY_BG"], foreground=t["FG"],
                        arrowcolor=t["FG"], padding=3)
        style.configure("TCheckbutton", background=t["BG"], foreground=t["FG"])
        style.configure("TNotebook", background=t["BG"], borderwidth=0)
        style.configure("TNotebook.Tab", background=t["SURFACE0"],
                        foreground=t["FG"],
                        padding=[14, 5], font=_font_bold)
        style.map("TNotebook.Tab",
                  background=[("selected", t["ACCENT"])],
                  foreground=[("selected", "#ffffff")])
        style.configure("Result.TLabel", font=("Helvetica", 32, "bold"),
                        foreground=t["ACCENT2"], background=t["BG"])
        style.configure("ResultUnit.TLabel", font=("Helvetica", 13),
                        foreground=t.get("FG_DIM", t["FG"]), background=t["BG"])
        style.configure("Preset.TButton", font=_font_sm, padding=(6, 3),
                        background=t["SURFACE0"], foreground=t["FG"])
        style.map("Preset.TButton",
                  background=[("active", t["ACCENT"])],
                  foreground=[("active", "#ffffff")])
        style.configure("Small.TLabel", font=_font_sm,
                        foreground=t.get("FG_DIM", t["BORDER"]),
                        background=t["BG"])
        style.configure("Error.TLabel", font=_font_sm,
                        foreground=t["ERROR_FG"], background=t["BG"])
        style.configure("Treeview", background=t["ENTRY_BG"], foreground=t["FG"],
                        fieldbackground=t["ENTRY_BG"], font=_font_sm,
                        rowheight=24)
        style.configure("Treeview.Heading", background=t["SURFACE0"],
                        foreground=t["FG"], font=_font_bold)
        style.configure("Heading.TLabel", font=("Helvetica", 13, "bold"),
                        foreground=t["ACCENT"], background=t["BG"])
        style.configure("TSeparator", background=t["BORDER"])
        style.configure("Card.TFrame", background=t["BG_LIGHT"])

    def _toggle_theme(self) -> None:
        self._theme = LIGHT if self._theme is DARK else DARK
        self._apply_theme()
        self._theme_btn.configure(
            text="Dark Mode" if self._theme is LIGHT else "Light Mode"
        )
        self._run_simulation()

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
        self.var_ftp = tk.DoubleVar(value=250)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_main_tab()
        self._build_course_tab()
        self._build_workout_tab()
        self._build_compare_tab()
        self._build_pw_ratio_tab()
        self._build_race_tab()
        self._build_3d_tab()
        self._build_cp_tab()
        self._build_pacing_tab()
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
        self._add_slider(frame, "Power (W)", self.var_power, 0, 1500, 1)
        self._add_slider(frame, "Weight (kg)", self.var_rider_weight, 30, 150, 0.5)
        self._add_slider(frame, "Height (cm)", self.var_rider_height, 140, 210, 1)
        self._add_slider(frame, "FTP (W)", self.var_ftp, 50, 500, 5)

    def _build_bike_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Bike")
        frame.pack(fill=tk.X, pady=(0, 6))
        self._add_slider(frame, "Bike Weight (kg)", self.var_bike_weight, 3, 25, 0.1)

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

        self._add_slider(frame, "Drivetrain Eff. (%)", self.var_efficiency, 85, 100, 0.5)
        self._add_slider(frame, "Wheel Mass (kg)", self.var_wheel_mass, 0.5, 4.0, 0.1)

    def _build_course_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Course / Environment")
        frame.pack(fill=tk.X, pady=(0, 6))
        self._add_slider(frame, "Grade (%)", self.var_grade, -20, 25, 0.1)
        self._add_slider(frame, "Wind Speed (km/h)", self.var_wind, 0, 80, 1)
        self._add_slider(frame, "Wind Dir (0=head 180=tail)", self.var_wind_dir, 0, 360, 5)
        self._add_slider(frame, "Elevation (m)", self.var_elevation, 0, 5000, 10)
        self._add_slider(frame, "Temperature (\u00b0C)", self.var_temperature, -10, 50, 1)

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
        self._theme_btn = ttk.Button(frame2, text="Light Mode",
                                      command=self._toggle_theme)
        self._theme_btn.pack(side=tk.LEFT, padx=4)

    def _build_results(self, parent: tk.Widget) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 4))

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

        # Calories display
        self.lbl_calories = ttk.Label(frame, text="", style="Small.TLabel")
        self.lbl_calories.pack(side=tk.RIGHT, padx=8)

        # W/kg display
        self.lbl_wkg = ttk.Label(frame, text="", style="Small.TLabel")
        self.lbl_wkg.pack(side=tk.RIGHT, padx=8)

        # Power zone display
        self.lbl_zone = ttk.Label(frame, text="", style="Small.TLabel")
        self.lbl_zone.pack(side=tk.RIGHT, padx=8)

        detail_frame = ttk.Frame(parent)
        detail_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.lbl_details = ttk.Label(detail_frame, text="", style="Small.TLabel",
                                     wraplength=600, justify=tk.LEFT)
        self.lbl_details.pack(anchor=tk.W)

    def _build_charts(self, parent: tk.Widget) -> None:
        t = self._theme
        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor=t["BG"])
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

        seg_frame = ttk.LabelFrame(left, text="Course Segments")
        seg_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        btn_row = ttk.Frame(seg_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_row, text="Add Segment", style="Preset.TButton",
                   command=self._add_course_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Remove Last", style="Preset.TButton",
                   command=self._remove_course_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Move Up", style="Preset.TButton",
                   command=self._move_segment_up).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Move Down", style="Preset.TButton",
                   command=self._move_segment_down).pack(side=tk.LEFT, padx=2)

        btn_row2 = ttk.Frame(seg_frame)
        btn_row2.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(btn_row2, text="Import GPX", style="Preset.TButton",
                   command=self._import_gpx).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row2, text="Run Profile",
                   command=self._run_course_profile).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_row2, text="Animate", style="Preset.TButton",
                   command=self._animate_course).pack(side=tk.RIGHT, padx=2)

        # Segment listbox for selection (for move up/down)
        self._seg_listbox = tk.Listbox(seg_frame, bg=self._theme["ENTRY_BG"],
                                        fg=self._theme["FG"], height=12,
                                        selectmode=tk.SINGLE, font=("Segoe UI", 9))
        self._seg_listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Add default segments
        for grade in [0, 5, -3]:
            self._add_course_segment(grade=grade)

        self.lbl_course_summary = ttk.Label(left, text="", style="Small.TLabel",
                                             wraplength=300, justify=tk.LEFT)
        self.lbl_course_summary.pack(fill=tk.X, pady=4)

        t = self._theme
        self.fig_course = Figure(figsize=(7, 5), dpi=100, facecolor=t["BG"])
        self.fig_course.subplots_adjust(hspace=0.40, left=0.10, right=0.95,
                                         top=0.94, bottom=0.10)
        self.ax_course_speed = self.fig_course.add_subplot(211)
        self.ax_course_elev = self.fig_course.add_subplot(212)
        self.canvas_course = FigureCanvasTkAgg(self.fig_course, master=right)
        self.canvas_course.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_course_segment(self, grade: float = 0.0, distance: float = 1000.0,
                             wind: float = 0.0, elev: float = 100.0,
                             temp: float = 20.0) -> None:
        seg_vars: dict[str, tk.DoubleVar] = {
            "distance": tk.DoubleVar(value=distance),
            "grade": tk.DoubleVar(value=grade),
            "wind": tk.DoubleVar(value=wind),
            "wind_dir": tk.DoubleVar(value=0),
            "elevation": tk.DoubleVar(value=elev),
            "temperature": tk.DoubleVar(value=temp),
        }
        self._course_segments.append(seg_vars)
        self._refresh_seg_listbox()

    def _remove_course_segment(self) -> None:
        sel = self._seg_listbox.curselection()
        if sel:
            idx = sel[0]
            self._course_segments.pop(idx)
        elif self._course_segments:
            self._course_segments.pop()
        self._refresh_seg_listbox()

    def _move_segment_up(self) -> None:
        sel = self._seg_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        idx = sel[0]
        self._course_segments[idx], self._course_segments[idx - 1] = \
            self._course_segments[idx - 1], self._course_segments[idx]
        self._refresh_seg_listbox()
        self._seg_listbox.selection_set(idx - 1)

    def _move_segment_down(self) -> None:
        sel = self._seg_listbox.curselection()
        if not sel or sel[0] >= len(self._course_segments) - 1:
            return
        idx = sel[0]
        self._course_segments[idx], self._course_segments[idx + 1] = \
            self._course_segments[idx + 1], self._course_segments[idx]
        self._refresh_seg_listbox()
        self._seg_listbox.selection_set(idx + 1)

    def _refresh_seg_listbox(self) -> None:
        self._seg_listbox.delete(0, tk.END)
        for i, sv in enumerate(self._course_segments):
            d = sv["distance"].get()
            g = sv["grade"].get()
            self._seg_listbox.insert(tk.END, f"Seg {i+1}: {d:.0f}m @ {g:.1f}%")

    def _import_gpx(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("GPX files", "*.gpx")])
        if not path:
            return
        try:
            segments = parse_gpx(path)
            if not segments:
                messagebox.showwarning("GPX", "No trackpoints found in file.")
                return
            self._course_segments.clear()
            for seg in segments:
                self._add_course_segment(
                    grade=seg.grade_pct, distance=seg.distance_m,
                    elev=seg.elevation_m, temp=seg.temperature_c)
            messagebox.showinfo("GPX", f"Imported {len(segments)} segments from GPX.")
        except Exception as e:
            messagebox.showerror("GPX Error", str(e))

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
                messagebox.showerror("Input Error", "Invalid values in segment")
                return

        if not segments:
            messagebox.showwarning("No Segments", "Add at least one course segment.")
            return

        result = simulate_course_profile(rider, bike, segments)
        self._last_course_result = result

        total_cal = sum(
            estimate_calories(rider.power_watts, seg.time_s)
            for seg in result.segments
        )

        mins = int(result.total_time_s // 60)
        secs = int(result.total_time_s % 60)
        summary = (
            f"Total: {result.total_distance_m / 1000:.2f} km  |  "
            f"Time: {mins}:{secs:02d}\n"
            f"Avg Speed: {result.avg_speed_kmh:.1f} km/h\n"
            f"Elev Gain: {result.total_elevation_gain_m:.0f} m  |  "
            f"Loss: {result.total_elevation_loss_m:.0f} m\n"
            f"Est. Calories: {total_cal:.0f} kcal"
        )
        self.lbl_course_summary.configure(text=summary)
        self._update_course_charts(result)

    def _update_course_charts(self, result: Any) -> None:
        t = self._theme
        distances_km = [d / 1000 for d in result.distances_cumulative]

        ax1 = self.ax_course_speed
        ax1.clear()
        _style_ax(ax1, t)
        mid_distances = [(distances_km[i] + distances_km[i + 1]) / 2
                         for i in range(len(result.speeds))]
        ax1.bar(mid_distances, result.speeds, width=[
            distances_km[i + 1] - distances_km[i] for i in range(len(result.speeds))
        ], color=t["ACCENT"], alpha=0.8, edgecolor=t["BORDER"])
        ax1.axhline(result.avg_speed_kmh, color=t["ACCENT4"], linestyle="--",
                    linewidth=1, label=f"Avg: {result.avg_speed_kmh:.1f} km/h")
        ax1.set_xlabel("Distance (km)", color=t["FG"], fontsize=9)
        ax1.set_ylabel("Speed (km/h)", color=t["FG"], fontsize=9)
        ax1.set_title("Speed by Segment", color=t["FG"], fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"])

        ax2 = self.ax_course_elev
        ax2.clear()
        _style_ax(ax2, t)
        ax2.fill_between(distances_km, result.elevations, alpha=0.3, color=t["ACCENT2"])
        ax2.plot(distances_km, result.elevations, color=t["ACCENT2"], linewidth=2)
        ax2.set_xlabel("Distance (km)", color=t["FG"], fontsize=9)
        ax2.set_ylabel("Elevation (m)", color=t["FG"], fontsize=9)
        ax2.set_title("Elevation Profile", color=t["FG"], fontsize=11, fontweight="bold")

        self.canvas_course.draw_idle()

    def _animate_course(self) -> None:
        if not self._last_course_result:
            messagebox.showwarning("No Data", "Run a course profile first.")
            return
        result = self._last_course_result
        distances_km = [d / 1000 for d in result.distances_cumulative]
        elevations = result.elevations

        self._update_course_charts(result)

        ax2 = self.ax_course_elev
        dot, = ax2.plot([], [], "o", color=self._theme["ACCENT4"], markersize=10, zorder=10)
        self.canvas_course.draw()

        total_frames = len(distances_km)
        frame_idx = [0]

        def step() -> None:
            i = frame_idx[0]
            if i >= total_frames:
                return
            dot.set_data([distances_km[i]], [elevations[i]])
            self.canvas_course.draw_idle()
            frame_idx[0] += 1
            self._animation_id = self.after(100, step)

        step()

    # ==================================================================
    # TAB 3: Workout / Intervals
    # ==================================================================
    def _build_workout_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  Workout  ")

        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=4, expand=False)

        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=4)

        intv_frame = ttk.LabelFrame(left, text="Interval Steps")
        intv_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        btn_row = ttk.Frame(intv_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_row, text="Add Step", style="Preset.TButton",
                   command=self._add_interval_step).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Remove Last", style="Preset.TButton",
                   command=self._remove_interval_step).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Run Workout",
                   command=self._run_workout).pack(side=tk.RIGHT, padx=2)

        self._intv_listbox = tk.Listbox(intv_frame, bg=self._theme["ENTRY_BG"],
                                         fg=self._theme["FG"], height=8,
                                         font=("Segoe UI", 9))
        self._intv_listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Default workout: warmup / interval / recovery / interval / cooldown
        for pw, dur in [(150, 300), (300, 180), (150, 120), (350, 180), (100, 300)]:
            self._add_interval_step(power=pw, duration=dur)

        self.lbl_workout_summary = ttk.Label(left, text="", style="Small.TLabel",
                                              wraplength=300, justify=tk.LEFT)
        self.lbl_workout_summary.pack(fill=tk.X, pady=4)

        t = self._theme
        self.fig_workout = Figure(figsize=(7, 5), dpi=100, facecolor=t["BG"])
        self.fig_workout.subplots_adjust(hspace=0.40, left=0.10, right=0.95,
                                          top=0.94, bottom=0.10)
        self.ax_workout_power = self.fig_workout.add_subplot(211)
        self.ax_workout_speed = self.fig_workout.add_subplot(212)
        self.canvas_workout = FigureCanvasTkAgg(self.fig_workout, master=right)
        self.canvas_workout.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_interval_step(self, power: float = 200, duration: float = 300) -> None:
        step_vars: dict[str, tk.DoubleVar] = {
            "power": tk.DoubleVar(value=power),
            "duration": tk.DoubleVar(value=duration),
        }
        self._interval_steps.append(step_vars)
        self._refresh_intv_listbox()

    def _remove_interval_step(self) -> None:
        if self._interval_steps:
            self._interval_steps.pop()
            self._refresh_intv_listbox()

    def _refresh_intv_listbox(self) -> None:
        self._intv_listbox.delete(0, tk.END)
        for i, sv in enumerate(self._interval_steps):
            pw = sv["power"].get()
            dur = sv["duration"].get()
            mins = int(dur // 60)
            secs = int(dur % 60)
            zone_name, _ = zone_for_power(pw, self.var_ftp.get())
            self._intv_listbox.insert(tk.END,
                f"Step {i+1}: {pw:.0f}W x {mins}:{secs:02d} ({zone_name})")

    def _run_workout(self) -> None:
        rider, bike, course = self._gather_params()
        steps: list[IntervalStep] = []
        for sv in self._interval_steps:
            steps.append(IntervalStep(
                power_watts=sv["power"].get(),
                duration_s=sv["duration"].get(),
            ))

        if not steps:
            messagebox.showwarning("No Steps", "Add at least one interval step.")
            return

        result = simulate_workout(steps, rider, bike, course, time_resolution_s=5.0)
        self._last_workout_result = result

        mins = int(result.total_time_s // 60)
        secs = int(result.total_time_s % 60)
        summary = (
            f"Total Time: {mins}:{secs:02d}\n"
            f"Distance: {result.total_distance_m / 1000:.2f} km\n"
            f"Avg Speed: {result.avg_speed_kmh:.1f} km/h\n"
            f"Avg Power: {result.avg_power_watts:.0f} W\n"
            f"Work: {result.total_work_kj:.0f} kJ\n"
            f"Calories: {result.total_calories_kcal:.0f} kcal"
        )
        self.lbl_workout_summary.configure(text=summary)
        self._update_workout_charts(result)

    def _update_workout_charts(self, result: Any) -> None:
        t = self._theme
        times_min = [pt.time_s / 60.0 for pt in result.points]
        powers = [pt.power_watts for pt in result.points]
        speeds = [pt.speed_kmh for pt in result.points]

        ftp = self.var_ftp.get()
        zones = power_zones(ftp)

        ax1 = self.ax_workout_power
        ax1.clear()
        _style_ax(ax1, t)

        # Zone background bands
        for name, lo, hi, color in zones:
            if hi > 3000:
                hi = max(powers) * 1.1 if powers else 500
            ax1.axhspan(lo, hi, alpha=0.10, color=color)

        # Color each point by zone
        for i in range(len(times_min) - 1):
            _, color = zone_for_power(powers[i], ftp)
            ax1.fill_between(
                [times_min[i], times_min[i + 1]],
                [powers[i], powers[i + 1]],
                alpha=0.6, color=color
            )
        ax1.plot(times_min, powers, color=t["FG"], linewidth=0.5, alpha=0.5)
        ax1.set_xlabel("Time (min)", color=t["FG"], fontsize=9)
        ax1.set_ylabel("Power (W)", color=t["FG"], fontsize=9)
        ax1.set_title("Workout Power Profile", color=t["FG"], fontsize=11, fontweight="bold")

        ax2 = self.ax_workout_speed
        ax2.clear()
        _style_ax(ax2, t)
        ax2.plot(times_min, speeds, color=t["ACCENT"], linewidth=1.5)
        ax2.fill_between(times_min, speeds, alpha=0.2, color=t["ACCENT"])
        ax2.set_xlabel("Time (min)", color=t["FG"], fontsize=9)
        ax2.set_ylabel("Speed (km/h)", color=t["FG"], fontsize=9)
        ax2.set_title("Speed Over Time", color=t["FG"], fontsize=11, fontweight="bold")

        self.canvas_workout.draw_idle()

    # ==================================================================
    # TAB 4: Compare Scenarios
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

        cols = ("Name", "Power", "Weight", "Grade", "Wind", "Speed (km/h)", "Speed (mph)")
        self._compare_tree = ttk.Treeview(tab, columns=cols, show="headings", height=8)
        for col in cols:
            self._compare_tree.heading(col, text=col)
            self._compare_tree.column(col, width=100, anchor=tk.CENTER)
        self._compare_tree.pack(fill=tk.X, padx=8, pady=4)

        t = self._theme
        self.fig_compare = Figure(figsize=(7, 3), dpi=100, facecolor=t["BG"])
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
        t = self._theme
        ax = self.ax_compare
        ax.clear()
        _style_ax(ax, t)

        if not self._scenarios:
            self.canvas_compare.draw_idle()
            return

        chart_colors = [t[k] for k in CHART_COLOURS_KEYS]
        names = [s[0] for s in self._scenarios]
        speeds = [s[4].speed_kmh for s in self._scenarios]
        colors = [chart_colors[i % len(chart_colors)] for i in range(len(names))]

        bars = ax.bar(names, speeds, color=colors, edgecolor=t["BORDER"], alpha=0.85)
        for bar, spd in zip(bars, speeds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{spd:.1f}", ha="center", va="bottom", color=t["FG"], fontsize=8)

        ax.set_ylabel("Speed (km/h)", color=t["FG"], fontsize=9)
        ax.set_title("Scenario Comparison", color=t["FG"], fontsize=11, fontweight="bold")

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
    # TAB 5: Power-to-Weight Ratio Analysis
    # ==================================================================
    def _build_pw_ratio_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  W/kg Analysis  ")

        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=4)

        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=4)

        info_frame = ttk.LabelFrame(left, text="Your Power-to-Weight")
        info_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(info_frame, text="Analyze Current Settings",
                   command=self._run_pw_analysis).pack(fill=tk.X, padx=8, pady=4)

        self.lbl_pw_ratio = ttk.Label(info_frame, text="", style="Result.TLabel")
        self.lbl_pw_ratio.pack(padx=8, pady=4)
        self.lbl_pw_unit = ttk.Label(info_frame, text="W/kg", style="ResultUnit.TLabel")
        self.lbl_pw_unit.pack(padx=8)

        self.lbl_pw_category = ttk.Label(info_frame, text="",
                                          font=("Segoe UI", 14, "bold"))
        self.lbl_pw_category.pack(padx=8, pady=8)

        self.lbl_pw_details = ttk.Label(info_frame, text="", style="Small.TLabel",
                                         wraplength=280, justify=tk.LEFT)
        self.lbl_pw_details.pack(fill=tk.X, padx=8, pady=4)

        cat_frame = ttk.LabelFrame(left, text="Category Reference")
        cat_frame.pack(fill=tk.X, pady=(0, 6))

        ref_text = ""
        for name, lo, hi in RIDER_CATEGORIES:
            if hi == float("inf"):
                ref_text += f"  {name}: {lo:.1f}+ W/kg\n"
            else:
                ref_text += f"  {name}: {lo:.1f} - {hi:.1f} W/kg\n"
        ttk.Label(cat_frame, text=ref_text.strip(), style="Small.TLabel",
                  justify=tk.LEFT).pack(padx=8, pady=4)

        t = self._theme
        self.fig_pw = Figure(figsize=(7, 5), dpi=100, facecolor=t["BG"])
        self.fig_pw.subplots_adjust(hspace=0.40, left=0.12, right=0.95,
                                     top=0.94, bottom=0.10)
        self.ax_pw_bar = self.fig_pw.add_subplot(211)
        self.ax_pw_curve = self.fig_pw.add_subplot(212)
        self.canvas_pw = FigureCanvasTkAgg(self.fig_pw, master=right)
        self.canvas_pw.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _run_pw_analysis(self) -> None:
        ftp = self.var_ftp.get()
        weight = self.var_rider_weight.get()
        analysis = w_per_kg_analysis(ftp, weight)

        w_kg = analysis["w_per_kg"]
        category = analysis["category"]

        self.lbl_pw_ratio.configure(text=f"{w_kg:.2f}")
        self.lbl_pw_category.configure(text=category)

        details = (
            f"FTP: {ftp:.0f} W\n"
            f"Weight: {weight:.1f} kg\n"
            f"W/kg: {w_kg:.2f}\n\n"
            f"Category: {category}"
        )
        self.lbl_pw_details.configure(text=details)
        self._update_pw_charts(w_kg, ftp, weight)

    def _update_pw_charts(self, w_kg: float, ftp: float, weight: float) -> None:
        t = self._theme
        chart_colors = [t[k] for k in CHART_COLOURS_KEYS]

        ax1 = self.ax_pw_bar
        ax1.clear()
        _style_ax(ax1, t)

        cat_names = [c[0] for c in RIDER_CATEGORIES]
        cat_ranges = [(c[1], min(c[2], 8.0)) for c in RIDER_CATEGORIES]
        colors = []
        for c_name in cat_names:
            if c_name == classify_rider(w_kg):
                colors.append(t["ACCENT2"])
            else:
                colors.append(t["ACCENT"])

        cat_names_rev = list(reversed(cat_names))
        bottoms = [c[0] for c in reversed(cat_ranges)]
        heights = [c[1] - c[0] for c in reversed(cat_ranges)]
        colors_rev = list(reversed(colors))

        bars = ax1.barh(cat_names_rev, heights, left=bottoms,
                        color=colors_rev, edgecolor=t["BORDER"], alpha=0.8)
        ax1.axvline(w_kg, color=t["ACCENT4"], linewidth=2, linestyle="--",
                    label=f"You: {w_kg:.2f} W/kg")
        ax1.set_xlabel("W/kg", color=t["FG"], fontsize=9)
        ax1.set_title("Rider Classification", color=t["FG"],
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"])

        ax2 = self.ax_pw_curve
        ax2.clear()
        _style_ax(ax2, t)

        weights = list(range(50, 110))
        w_per_kgs = [ftp / w for w in weights]
        ax2.plot(weights, w_per_kgs, color=t["ACCENT"], linewidth=2)
        ax2.axvline(weight, color=t["ACCENT4"], linestyle="--", linewidth=1,
                    label=f"Your weight: {weight:.0f} kg")
        ax2.axhline(w_kg, color=t["ACCENT2"], linestyle=":", linewidth=1,
                    alpha=0.7, label=f"Your W/kg: {w_kg:.2f}")

        for name, lo, hi in RIDER_CATEGORIES:
            if hi == float("inf"):
                hi = 8.0
            ax2.axhspan(lo, hi, alpha=0.06, color=t["ACCENT3"])

        ax2.fill_between(weights, w_per_kgs, alpha=0.15, color=t["ACCENT"])
        ax2.set_xlabel("Rider Weight (kg)", color=t["FG"], fontsize=9)
        ax2.set_ylabel("W/kg at current FTP", color=t["FG"], fontsize=9)
        ax2.set_title(f"W/kg vs Weight (FTP={ftp:.0f}W)", color=t["FG"],
                      fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"])

        self.canvas_pw.draw_idle()

    # ==================================================================
    # TAB 7: Race Simulation
    # ==================================================================
    def _build_race_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  Race  ")

        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=4)

        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=4)

        comp_frame = ttk.LabelFrame(left, text="Competitors")
        comp_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        btn_row = ttk.Frame(comp_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_row, text="Add Competitor", style="Preset.TButton",
                   command=self._add_race_competitor).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Add Me (Current)", style="Preset.TButton",
                   command=self._add_me_as_competitor).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Remove Selected", style="Preset.TButton",
                   command=self._remove_race_competitor).pack(side=tk.LEFT, padx=2)

        self._race_competitors: list[RaceCompetitor] = []
        self._race_listbox = tk.Listbox(comp_frame, bg=self._theme["ENTRY_BG"],
                                         fg=self._theme["FG"], height=8,
                                         selectmode=tk.SINGLE,
                                         font=("Segoe UI", 9))
        self._race_listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Pre-populate with virtual competitors
        self._race_competitors = [
            RaceCompetitor("You", 250, 75, 178, 8.0),
            RaceCompetitor("Strong Rider", 300, 72, 175, 7.5,
                           position=RidingPosition.DROPS),
            RaceCompetitor("TT Specialist", 320, 78, 182, 8.0,
                           position=RidingPosition.AEROBARS),
            RaceCompetitor("Climber", 260, 60, 170, 6.5,
                           tire_type=TireType.ROAD_RACING),
        ]
        self._refresh_race_listbox()

        btn_row2 = ttk.Frame(comp_frame)
        btn_row2.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_row2, text="Run Race",
                   command=self._run_race).pack(side=tk.RIGHT, padx=2)

        self.lbl_race_results = ttk.Label(left, text="", style="Small.TLabel",
                                           wraplength=300, justify=tk.LEFT)
        self.lbl_race_results.pack(fill=tk.X, pady=4)

        t = self._theme
        self.fig_race = Figure(figsize=(7, 5), dpi=100, facecolor=t["BG"])
        self.fig_race.subplots_adjust(hspace=0.40, left=0.10, right=0.95,
                                       top=0.94, bottom=0.14)
        self.ax_race_speed = self.fig_race.add_subplot(211)
        self.ax_race_time = self.fig_race.add_subplot(212)
        self.canvas_race = FigureCanvasTkAgg(self.fig_race, master=right)
        self.canvas_race.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _refresh_race_listbox(self) -> None:
        self._race_listbox.delete(0, tk.END)
        for c in self._race_competitors:
            w_kg = power_to_weight_ratio(c.power_watts, c.weight_kg)
            self._race_listbox.insert(
                tk.END,
                f"{c.name}: {c.power_watts:.0f}W / {c.weight_kg:.0f}kg "
                f"({w_kg:.1f} W/kg)"
            )

    def _add_race_competitor(self) -> None:
        name = simpledialog.askstring("Competitor", "Name:", parent=self)
        if not name:
            return
        power = simpledialog.askfloat("Competitor", "Power (W):",
                                       initialvalue=250, parent=self)
        weight = simpledialog.askfloat("Competitor", "Weight (kg):",
                                        initialvalue=75, parent=self)
        if power is None or weight is None:
            return
        self._race_competitors.append(
            RaceCompetitor(name, power, weight)
        )
        self._refresh_race_listbox()

    def _add_me_as_competitor(self) -> None:
        rider, bike, _ = self._gather_params()
        ftp = self.var_ftp.get()
        comp = RaceCompetitor(
            name="You",
            power_watts=ftp,
            weight_kg=rider.weight_kg,
            height_cm=rider.height_cm,
            bike_weight_kg=bike.weight_kg,
            tire_type=bike.tire_type,
            position=bike.position,
            drivetrain_efficiency=bike.drivetrain_efficiency,
        )
        self._race_competitors.append(comp)
        self._refresh_race_listbox()

    def _remove_race_competitor(self) -> None:
        sel = self._race_listbox.curselection()
        if sel:
            self._race_competitors.pop(sel[0])
            self._refresh_race_listbox()

    def _run_race(self) -> None:
        if not self._race_competitors:
            messagebox.showwarning("No Competitors", "Add at least one competitor.")
            return

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
                pass

        if not segments:
            segments = [
                CourseSegment(distance_m=5000, grade_pct=0),
                CourseSegment(distance_m=2000, grade_pct=5),
                CourseSegment(distance_m=3000, grade_pct=-3),
                CourseSegment(distance_m=5000, grade_pct=0),
                CourseSegment(distance_m=1000, grade_pct=8),
                CourseSegment(distance_m=4000, grade_pct=-2),
            ]

        race_results = simulate_race(self._race_competitors, segments)

        result_text = "Race Results:\n"
        for i, r in enumerate(race_results):
            mins = int(r.total_time_s // 60)
            secs = int(r.total_time_s % 60)
            gap = f"+{r.gap_to_leader_s:.1f}s" if r.gap_to_leader_s > 0 else "Leader"
            result_text += (
                f"  {i+1}. {r.name}: {mins}:{secs:02d} "
                f"({r.avg_speed_kmh:.1f} km/h) {gap}\n"
            )
        self.lbl_race_results.configure(text=result_text.strip())
        self._update_race_charts(race_results, segments)

    def _update_race_charts(self, race_results: list, segments: list) -> None:
        t = self._theme
        chart_colors = [t[k] for k in CHART_COLOURS_KEYS]

        ax1 = self.ax_race_speed
        ax1.clear()
        _style_ax(ax1, t)

        total_dist = sum(s.distance_m for s in segments)
        for i, r in enumerate(race_results):
            color = chart_colors[i % len(chart_colors)]
            mid_d = []
            cum = 0.0
            for j, seg in enumerate(segments):
                mid_d.append((cum + seg.distance_m / 2) / 1000)
                cum += seg.distance_m
            ax1.plot(mid_d, r.segment_speeds, color=color, linewidth=1.5,
                     marker="o", markersize=4, label=r.name)

        ax1.set_xlabel("Distance (km)", color=t["FG"], fontsize=9)
        ax1.set_ylabel("Speed (km/h)", color=t["FG"], fontsize=9)
        ax1.set_title("Speed by Segment", color=t["FG"], fontsize=11,
                      fontweight="bold")
        ax1.legend(fontsize=7, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"], loc="best")

        ax2 = self.ax_race_time
        ax2.clear()
        _style_ax(ax2, t)

        names = [r.name for r in race_results]
        times = [r.total_time_s for r in race_results]
        colors = [chart_colors[i % len(chart_colors)]
                  for i in range(len(race_results))]

        bars = ax2.barh(names, times, color=colors, edgecolor=t["BORDER"],
                        alpha=0.85)
        for bar, r in zip(bars, race_results):
            mins = int(r.total_time_s // 60)
            secs = int(r.total_time_s % 60)
            gap_txt = (f" (+{r.gap_to_leader_s:.1f}s)"
                       if r.gap_to_leader_s > 0 else " (Leader)")
            ax2.text(bar.get_width() + max(times) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"{mins}:{secs:02d}{gap_txt}",
                     va="center", color=t["FG"], fontsize=8)

        ax2.set_xlabel("Time (s)", color=t["FG"], fontsize=9)
        ax2.set_title("Race Standings", color=t["FG"], fontsize=11,
                      fontweight="bold")

        self.canvas_race.draw_idle()

    # ==================================================================
    # TAB 7: 3D Course Visualization & Gradient Map
    # ==================================================================
    def _build_3d_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  3D / Map  ")

        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=8, pady=4)

        ttk.Button(top, text="Show 3D Course",
                   command=self._show_3d_course).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Show Elevation Heatmap",
                   command=self._show_elevation_heatmap).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Load GPX for Gradient Map",
                   command=self._show_gradient_map).pack(side=tk.LEFT, padx=4)

        t = self._theme
        self.fig_3d = Figure(figsize=(12, 5), dpi=100, facecolor=t["BG"])
        self.fig_3d.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08,
                                     wspace=0.25)
        self.ax_3d = self.fig_3d.add_subplot(121, projection="3d")
        self.ax_gradient_map = self.fig_3d.add_subplot(122)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=tab)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8)

    def _show_3d_course(self) -> None:
        if not self._last_course_result:
            messagebox.showwarning("No Data", "Run a course profile first.")
            return

        result = self._last_course_result
        t = self._theme

        distances_km = [d / 1000 for d in result.distances_cumulative]
        elevations = result.elevations

        ax = self.ax_3d
        ax.clear()

        ax.set_facecolor(t["BG_LIGHT"])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        n = len(distances_km)
        x = distances_km
        y = [0.0] * n
        z = elevations

        for i in range(n - 1):
            grade = 0.0
            dist_m = (x[i + 1] - x[i]) * 1000
            if dist_m > 0:
                grade = (z[i + 1] - z[i]) / dist_m * 100

            if grade > 5:
                color = "#f38ba8"
            elif grade > 2:
                color = "#fab387"
            elif grade > 0:
                color = "#f9e2af"
            elif grade > -2:
                color = "#a6e3a1"
            else:
                color = "#89b4fa"

            ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]],
                    color=color, linewidth=3)

        if len(x) > 0:
            ax.scatter([x[0]], [y[0]], [z[0]], color=t["ACCENT2"], s=80,
                       marker="o", zorder=10, label="Start")
            ax.scatter([x[-1]], [y[-1]], [z[-1]], color=t["ACCENT4"], s=80,
                       marker="s", zorder=10, label="Finish")

        width = max(0.02, (max(x) - min(x)) * 0.15) if len(x) > 1 else 0.1
        for i in range(n):
            ax.plot([x[i], x[i]], [-width, width], [z[i], z[i]],
                    color=t["ACCENT2"], alpha=0.15, linewidth=0.5)

        ax.set_xlabel("Distance (km)", color=t["FG"], fontsize=8, labelpad=8)
        ax.set_ylabel("", color=t["FG"], fontsize=8)
        ax.set_zlabel("Elevation (m)", color=t["FG"], fontsize=8, labelpad=8)
        ax.set_title("3D Course Profile", color=t["FG"], fontsize=11,
                     fontweight="bold")
        ax.tick_params(colors=t["FG"], labelsize=7)
        ax.set_yticks([])
        ax.legend(fontsize=7, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                  labelcolor=t["FG"])

        ax.view_init(elev=25, azim=-60)
        self.canvas_3d.draw_idle()

    def _show_elevation_heatmap(self) -> None:
        if not self._last_course_result:
            messagebox.showwarning("No Data", "Run a course profile first.")
            return

        result = self._last_course_result
        t = self._theme

        distances_km = [d / 1000 for d in result.distances_cumulative]
        elevations = result.elevations

        ax = self.ax_3d
        ax.clear()
        ax.set_facecolor(t["BG_LIGHT"])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        n = len(distances_km)
        x = distances_km
        z = elevations

        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "elev_heat",
            [(0.0, "#89b4fa"), (0.25, "#a6e3a1"), (0.5, "#f9e2af"),
             (0.75, "#fab387"), (1.0, "#f38ba8")]
        )

        min_ele = min(z) if z else 0
        max_ele = max(z) if z else 100
        ele_range = max(max_ele - min_ele, 1.0)

        width = max(0.05, (max(x) - min(x)) * 0.2) if len(x) > 1 else 0.1
        y_vals = list(range(5))
        y_norm = [v / 4.0 * width * 2 - width for v in y_vals]

        for i in range(n - 1):
            for j in range(len(y_norm) - 1):
                xs = [x[i], x[i+1], x[i+1], x[i]]
                ys = [y_norm[j], y_norm[j], y_norm[j+1], y_norm[j+1]]
                avg_ele = (z[i] + z[i+1]) / 2
                norm_val = (avg_ele - min_ele) / ele_range
                color = cmap(norm_val)
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                verts = [[
                    (xs[0], ys[0], z[i]),
                    (xs[1], ys[1], z[i+1]),
                    (xs[2], ys[2], z[i+1]),
                    (xs[3], ys[3], z[i]),
                ]]
                poly = Poly3DCollection(verts, alpha=0.7,
                                        facecolor=color, edgecolor=color)
                ax.add_collection3d(poly)

        if len(x) > 0:
            ax.scatter([x[0]], [0], [z[0]], color=t["ACCENT2"], s=100,
                       marker="o", zorder=10, label="Start")
            ax.scatter([x[-1]], [0], [z[-1]], color=t["ACCENT4"], s=100,
                       marker="s", zorder=10, label="Finish")

        ax.set_xlabel("Distance (km)", color=t["FG"], fontsize=8, labelpad=8)
        ax.set_ylabel("", color=t["FG"], fontsize=8)
        ax.set_zlabel("Elevation (m)", color=t["FG"], fontsize=8, labelpad=8)
        ax.set_title("Elevation Heatmap Overlay", color=t["FG"], fontsize=11,
                     fontweight="bold")
        ax.tick_params(colors=t["FG"], labelsize=7)
        ax.set_yticks([])
        ax.legend(fontsize=7, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                  labelcolor=t["FG"])
        ax.view_init(elev=30, azim=-65)

        self.canvas_3d.draw_idle()

    def _show_gradient_map(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("GPX files", "*.gpx")])
        if not path:
            return
        try:
            points = parse_gpx_with_coords(path)
            if not points:
                messagebox.showwarning("GPX", "No trackpoints found.")
                return
            self._draw_gradient_map(points)
        except Exception as e:
            messagebox.showerror("GPX Error", str(e))

    def _draw_gradient_map(self, points: list[dict]) -> None:
        t = self._theme
        ax = self.ax_gradient_map
        ax.clear()
        _style_ax(ax, t)

        lats = [p["lat"] for p in points]
        lons = [p["lon"] for p in points]
        grades = [p["grade_pct"] for p in points]

        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "grade",
            [(0.0, "#89b4fa"), (0.25, "#a6e3a1"), (0.5, "#f9e2af"),
             (0.75, "#fab387"), (1.0, "#f38ba8")]
        )

        max_abs_grade = max(abs(g) for g in grades) if grades else 10
        max_abs_grade = max(max_abs_grade, 1.0)

        for i in range(len(lons) - 1):
            normalized = (grades[i] + max_abs_grade) / (2 * max_abs_grade)
            normalized = max(0.0, min(1.0, normalized))
            color = cmap(normalized)
            ax.plot([lons[i], lons[i + 1]], [lats[i], lats[i + 1]],
                    color=color, linewidth=3, solid_capstyle="round")

        ax.scatter([lons[0]], [lats[0]], color=t["ACCENT2"], s=60,
                   marker="o", zorder=10, label="Start")
        ax.scatter([lons[-1]], [lats[-1]], color=t["ACCENT4"], s=60,
                   marker="s", zorder=10, label="Finish")

        sm = matplotlib.cm.ScalarMappable(
            cmap=cmap,
            norm=matplotlib.colors.Normalize(vmin=-max_abs_grade,
                                             vmax=max_abs_grade)
        )
        sm.set_array([])
        cbar = self.fig_3d.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cbar.set_label("Grade (%)", color=t["FG"], fontsize=9)
        cbar.ax.tick_params(colors=t["FG"], labelsize=7)

        ax.set_xlabel("Longitude", color=t["FG"], fontsize=9)
        ax.set_ylabel("Latitude", color=t["FG"], fontsize=9)
        ax.set_title("Gradient-Colored Route Map", color=t["FG"],
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                  labelcolor=t["FG"])
        ax.set_aspect("equal")

        self.canvas_3d.draw_idle()

    # ==================================================================
    # TAB 8: Critical Power Model (CP & W')
    # ==================================================================
    def _build_cp_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  CP / W'  ")

        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=4)

        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=4)

        est_frame = ttk.LabelFrame(left, text="Estimate from FTP")
        est_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(est_frame, text="Uses your current FTP slider value",
                  style="Small.TLabel").pack(padx=8, pady=2)
        ttk.Button(est_frame, text="Estimate CP from FTP",
                   command=self._estimate_cp).pack(fill=tk.X, padx=8, pady=4)

        fit_frame = ttk.LabelFrame(left, text="Fit from Two Efforts")
        fit_frame.pack(fill=tk.X, pady=(0, 6))

        self.var_cp_power1 = tk.DoubleVar(value=350)
        self.var_cp_dur1 = tk.DoubleVar(value=300)
        self.var_cp_power2 = tk.DoubleVar(value=280)
        self.var_cp_dur2 = tk.DoubleVar(value=1200)

        r1 = ttk.Frame(fit_frame)
        r1.pack(fill=tk.X, padx=8, pady=2)
        _label(r1, "Effort 1: Power (W)").pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.var_cp_power1, width=6).pack(side=tk.RIGHT)

        r2 = ttk.Frame(fit_frame)
        r2.pack(fill=tk.X, padx=8, pady=2)
        _label(r2, "Effort 1: Duration (s)").pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self.var_cp_dur1, width=6).pack(side=tk.RIGHT)

        r3 = ttk.Frame(fit_frame)
        r3.pack(fill=tk.X, padx=8, pady=2)
        _label(r3, "Effort 2: Power (W)").pack(side=tk.LEFT)
        ttk.Entry(r3, textvariable=self.var_cp_power2, width=6).pack(side=tk.RIGHT)

        r4 = ttk.Frame(fit_frame)
        r4.pack(fill=tk.X, padx=8, pady=2)
        _label(r4, "Effort 2: Duration (s)").pack(side=tk.LEFT)
        ttk.Entry(r4, textvariable=self.var_cp_dur2, width=6).pack(side=tk.RIGHT)

        ttk.Button(fit_frame, text="Fit CP Model",
                   command=self._fit_cp).pack(fill=tk.X, padx=8, pady=4)

        res_frame = ttk.LabelFrame(left, text="Results")
        res_frame.pack(fill=tk.X, pady=(0, 6))
        self.lbl_cp_result = ttk.Label(res_frame, text="", style="Small.TLabel",
                                        wraplength=280, justify=tk.LEFT)
        self.lbl_cp_result.pack(padx=8, pady=4)

        t = self._theme
        self.fig_cp = Figure(figsize=(7, 5), dpi=100, facecolor=t["BG"])
        self.fig_cp.subplots_adjust(hspace=0.40, left=0.12, right=0.95,
                                     top=0.94, bottom=0.12)
        self.ax_cp_curve = self.fig_cp.add_subplot(211)
        self.ax_cp_tte = self.fig_cp.add_subplot(212)
        self.canvas_cp = FigureCanvasTkAgg(self.fig_cp, master=right)
        self.canvas_cp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _estimate_cp(self) -> None:
        ftp = self.var_ftp.get()
        model = estimate_cp_from_ftp(ftp)
        self._display_cp_model(model)

    def _fit_cp(self) -> None:
        try:
            model = fit_cp_model(
                self.var_cp_power1.get(), self.var_cp_dur1.get(),
                self.var_cp_power2.get(), self.var_cp_dur2.get(),
            )
            self._display_cp_model(model)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _display_cp_model(self, model: CriticalPowerModel) -> None:
        t = self._theme

        tte_300 = model.time_to_exhaustion(300)
        tte_400 = model.time_to_exhaustion(400)
        tte_str = lambda t_s: f"{t_s:.0f}s" if t_s < float("inf") else "unlimited"

        result_text = (
            f"Critical Power (CP): {model.cp_watts:.0f} W\n"
            f"W' (anaerobic capacity): {model.w_prime_joules/1000:.1f} kJ\n\n"
            f"Max 1-min power: {model.max_power_for_duration(60):.0f} W\n"
            f"Max 5-min power: {model.max_power_for_duration(300):.0f} W\n"
            f"Max 20-min power: {model.max_power_for_duration(1200):.0f} W\n"
            f"Max 60-min power: {model.max_power_for_duration(3600):.0f} W\n\n"
            f"TTE at 300W: {tte_str(tte_300)}\n"
            f"TTE at 400W: {tte_str(tte_400)}"
        )
        self.lbl_cp_result.configure(text=result_text)

        durations, powers = model.power_curve()

        ax1 = self.ax_cp_curve
        ax1.clear()
        _style_ax(ax1, t)

        duration_mins = [d / 60 for d in durations]
        ax1.plot(duration_mins, powers, color=t["ACCENT"], linewidth=2.5,
                 marker="o", markersize=4)
        ax1.axhline(model.cp_watts, color=t["ACCENT4"], linestyle="--",
                    linewidth=1.5, label=f"CP = {model.cp_watts:.0f} W")
        ax1.fill_between(duration_mins, [model.cp_watts] * len(duration_mins),
                         powers, alpha=0.15, color=t["ACCENT3"],
                         label=f"W' = {model.w_prime_joules/1000:.1f} kJ")
        ax1.set_xlabel("Duration (min)", color=t["FG"], fontsize=9)
        ax1.set_ylabel("Power (W)", color=t["FG"], fontsize=9)
        ax1.set_title("Power-Duration Curve", color=t["FG"],
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"])
        ax1.set_xscale("log")

        ax2 = self.ax_cp_tte
        ax2.clear()
        _style_ax(ax2, t)

        test_powers = list(range(int(model.cp_watts) + 10,
                                  int(model.cp_watts) + 250, 10))
        ttes = [model.time_to_exhaustion(p) / 60 for p in test_powers]

        ax2.plot(test_powers, ttes, color=t["ACCENT2"], linewidth=2)
        ax2.fill_between(test_powers, ttes, alpha=0.1, color=t["ACCENT2"])
        ax2.axvline(model.cp_watts, color=t["ACCENT4"], linestyle="--",
                    linewidth=1, alpha=0.7, label=f"CP = {model.cp_watts:.0f} W")
        ax2.set_xlabel("Power (W)", color=t["FG"], fontsize=9)
        ax2.set_ylabel("Time to Exhaustion (min)", color=t["FG"], fontsize=9)
        ax2.set_title("Time to Exhaustion", color=t["FG"],
                      fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"])

        self.canvas_cp.draw_idle()

    # ==================================================================
    # TAB 9: Route Optimization (Pacing Strategy)
    # ==================================================================
    def _build_pacing_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  Pacing  ")

        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=4)

        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=4)

        ctrl = ttk.LabelFrame(left, text="Pacing Settings")
        ctrl.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(ctrl, text="Uses course segments from Course Profile tab",
                  style="Small.TLabel").pack(padx=8, pady=2)

        self.var_target_power = tk.DoubleVar(value=200)
        r1 = ttk.Frame(ctrl)
        r1.pack(fill=tk.X, padx=8, pady=2)
        _label(r1, "Target Avg Power (W):").pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.var_target_power, width=6).pack(side=tk.RIGHT)

        ttk.Button(ctrl, text="Optimize Pacing",
                   command=self._run_pacing).pack(fill=tk.X, padx=8, pady=4)

        res_frame = ttk.LabelFrame(left, text="Optimization Results")
        res_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.lbl_pacing_result = ttk.Label(res_frame, text="", style="Small.TLabel",
                                            wraplength=280, justify=tk.LEFT)
        self.lbl_pacing_result.pack(padx=8, pady=4)

        t = self._theme
        self.fig_pacing = Figure(figsize=(7, 5), dpi=100, facecolor=t["BG"])
        self.fig_pacing.subplots_adjust(hspace=0.40, left=0.10, right=0.95,
                                         top=0.94, bottom=0.12)
        self.ax_pacing_power = self.fig_pacing.add_subplot(211)
        self.ax_pacing_speed = self.fig_pacing.add_subplot(212)
        self.canvas_pacing = FigureCanvasTkAgg(self.fig_pacing, master=right)
        self.canvas_pacing.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _run_pacing(self) -> None:
        rider, bike, _ = self._gather_params()
        ftp = self.var_ftp.get()

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
                pass

        if not segments:
            segments = [
                CourseSegment(distance_m=5000, grade_pct=0),
                CourseSegment(distance_m=2000, grade_pct=5),
                CourseSegment(distance_m=3000, grade_pct=-3),
                CourseSegment(distance_m=5000, grade_pct=0),
                CourseSegment(distance_m=1000, grade_pct=8),
                CourseSegment(distance_m=4000, grade_pct=-2),
            ]

        target = self.var_target_power.get()
        result = optimize_pacing(rider, bike, segments, ftp, target)

        mins = int(result.total_time_s // 60)
        secs = int(result.total_time_s % 60)
        even_mins = int(result.even_pace_time_s // 60)
        even_secs = int(result.even_pace_time_s % 60)

        text = (
            f"Optimized Time: {mins}:{secs:02d}\n"
            f"Even-Pace Time: {even_mins}:{even_secs:02d}\n"
            f"Time Saved: {result.time_saved_s:.1f}s\n"
            f"Avg Power: {result.avg_power:.0f} W\n"
            f"Avg Speed: {result.avg_speed_kmh:.1f} km/h\n\n"
            f"Segment Strategy:\n"
        )
        for ps in result.segments:
            text += (
                f"  Seg {ps.segment_index+1}: {ps.grade_pct:+.1f}% "
                f"→ {ps.optimal_power:.0f}W "
                f"({ps.speed_kmh:.1f} km/h) "
                f"[{ps.strategy_note}]\n"
            )

        self.lbl_pacing_result.configure(text=text.strip())
        self._update_pacing_charts(result)

    def _update_pacing_charts(self, result: Any) -> None:
        t = self._theme

        ax1 = self.ax_pacing_power
        ax1.clear()
        _style_ax(ax1, t)

        seg_indices = [ps.segment_index + 1 for ps in result.segments]
        powers = [ps.optimal_power for ps in result.segments]
        grades = [ps.grade_pct for ps in result.segments]

        colors = []
        for g in grades:
            if g > 5:
                colors.append(t["ACCENT4"])
            elif g > 2:
                colors.append(t["ACCENT3"])
            elif g > 0:
                colors.append(t["ACCENT2"])
            elif g > -2:
                colors.append(t["ACCENT"])
            else:
                colors.append("#74c7ec")

        bars = ax1.bar(seg_indices, powers, color=colors, edgecolor=t["BORDER"],
                       alpha=0.85)
        ax1.axhline(result.avg_power, color=t["ACCENT4"], linestyle="--",
                    linewidth=1.5, label=f"Avg: {result.avg_power:.0f} W")
        for bar, p, g in zip(bars, powers, grades):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 2,
                     f"{p:.0f}W\n{g:+.0f}%",
                     ha="center", va="bottom", color=t["FG"], fontsize=7)

        ax1.set_xlabel("Segment", color=t["FG"], fontsize=9)
        ax1.set_ylabel("Power (W)", color=t["FG"], fontsize=9)
        ax1.set_title("Optimal Power by Segment", color=t["FG"],
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"])

        ax2 = self.ax_pacing_speed
        ax2.clear()
        _style_ax(ax2, t)

        speeds = [ps.speed_kmh for ps in result.segments]
        ax2.bar(seg_indices, speeds, color=t["ACCENT"], edgecolor=t["BORDER"],
                alpha=0.85)
        ax2.axhline(result.avg_speed_kmh, color=t["ACCENT4"], linestyle="--",
                    linewidth=1.5, label=f"Avg: {result.avg_speed_kmh:.1f} km/h")
        for i, (idx, s) in enumerate(zip(seg_indices, speeds)):
            ax2.text(idx, s + 0.5, f"{s:.1f}",
                     ha="center", va="bottom", color=t["FG"], fontsize=7)

        ax2.set_xlabel("Segment", color=t["FG"], fontsize=9)
        ax2.set_ylabel("Speed (km/h)", color=t["FG"], fontsize=9)
        ax2.set_title("Resulting Speed by Segment", color=t["FG"],
                      fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"])

        self.canvas_pacing.draw_idle()

    # ==================================================================
    # TAB 10: Export
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
        ttk.Button(info, text="Export Workout to CSV",
                   command=self._export_workout_csv).pack(anchor=tk.W, pady=4)
        ttk.Button(info, text="Generate PDF Report",
                   command=self._export_pdf).pack(anchor=tk.W, pady=4)

    def _export_result_csv(self) -> None:
        rider, bike, course = self._gather_params()
        result = solve_speed(rider, bike, course)
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            export_result_csv(result, rider, bike, course, path)
            messagebox.showinfo("Exported", f"Results saved to {path}")

    def _export_course_csv(self) -> None:
        if not self._last_course_result:
            messagebox.showwarning("No Data", "Run a course profile first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            export_course_csv(self._last_course_result, path)
            messagebox.showinfo("Exported", f"Course profile saved to {path}")

    def _export_workout_csv(self) -> None:
        if not hasattr(self, "_last_workout_result"):
            messagebox.showwarning("No Data", "Run a workout first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            export_workout_csv(self._last_workout_result, path)
            messagebox.showinfo("Exported", f"Workout saved to {path}")

    def _export_pdf(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not path:
            return

        t = self._theme
        rider, bike, course = self._gather_params()
        result = solve_speed(rider, bike, course)

        with PdfPages(path) as pdf:
            # Page 1: Main simulation
            fig1 = Figure(figsize=(11, 8.5), dpi=150, facecolor="white")
            fig1.suptitle("Bike Power Simulator Report", fontsize=16, fontweight="bold")

            ax_info = fig1.add_subplot(311)
            ax_info.axis("off")
            info_text = (
                f"Power: {rider.power_watts:.0f} W  |  "
                f"Weight: {rider.weight_kg:.1f} kg  |  "
                f"Height: {rider.height_cm:.0f} cm\n"
                f"Bike: {bike.weight_kg:.1f} kg  |  "
                f"Tire: {bike.tire_type.value}  |  "
                f"Position: {bike.position.value}\n"
                f"Grade: {course.grade_pct:.1f}%  |  "
                f"Wind: {course.headwind_kmh:.0f} km/h\n\n"
                f"RESULT: {result.speed_kmh:.1f} km/h  "
                f"({result.speed_mph:.1f} mph)\n"
                f"Aero: {result.power_aero:.1f} W  |  "
                f"Rolling: {result.power_rolling:.1f} W  |  "
                f"Gravity: {result.power_gravity:.1f} W"
            )
            ax_info.text(0.05, 0.5, info_text, transform=ax_info.transAxes,
                        fontsize=10, verticalalignment="center", family="monospace")

            ax_curve = fig1.add_subplot(312)
            powers, speeds = speed_vs_power_curve(rider, bike, course)
            ax_curve.plot(powers, speeds, color="steelblue", linewidth=2)
            ax_curve.axvline(rider.power_watts, color="red", linestyle="--", linewidth=1)
            ax_curve.scatter([rider.power_watts], [result.speed_kmh], color="green", s=60, zorder=5)
            ax_curve.set_xlabel("Power (W)")
            ax_curve.set_ylabel("Speed (km/h)")
            ax_curve.set_title("Speed vs Power")
            ax_curve.grid(True, alpha=0.3)

            ax_pie = fig1.add_subplot(313)
            labels, sizes, colors = [], [], []
            for lbl, val, col in [("Aero", abs(result.power_aero), "steelblue"),
                                   ("Rolling", abs(result.power_rolling), "green"),
                                   ("Gravity", abs(result.power_gravity), "orange"),
                                   ("Drivetrain", abs(result.power_drivetrain_loss), "red")]:
                if val > 0.1:
                    labels.append(f"{lbl}\n{val:.0f} W")
                    sizes.append(val)
                    colors.append(col)
            if sizes:
                ax_pie.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90)
            ax_pie.set_title("Power Breakdown")

            fig1.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig1)

            # Page 2: Course profile (if available)
            if self._last_course_result:
                cr = self._last_course_result
                fig2 = Figure(figsize=(11, 8.5), dpi=150, facecolor="white")
                fig2.suptitle("Course Profile Report", fontsize=16, fontweight="bold")

                dk = [d / 1000 for d in cr.distances_cumulative]
                ax_s = fig2.add_subplot(211)
                mid_d = [(dk[i] + dk[i + 1]) / 2 for i in range(len(cr.speeds))]
                ax_s.bar(mid_d, cr.speeds, width=[dk[i + 1] - dk[i] for i in range(len(cr.speeds))],
                         color="steelblue", alpha=0.8)
                ax_s.axhline(cr.avg_speed_kmh, color="red", linestyle="--")
                ax_s.set_xlabel("Distance (km)")
                ax_s.set_ylabel("Speed (km/h)")
                ax_s.set_title("Speed by Segment")
                ax_s.grid(True, alpha=0.3)

                ax_e = fig2.add_subplot(212)
                ax_e.fill_between(dk, cr.elevations, alpha=0.3, color="green")
                ax_e.plot(dk, cr.elevations, color="green", linewidth=2)
                ax_e.set_xlabel("Distance (km)")
                ax_e.set_ylabel("Elevation (m)")
                ax_e.set_title("Elevation Profile")
                ax_e.grid(True, alpha=0.3)

                fig2.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig2)

        messagebox.showinfo("PDF Exported", f"Report saved to {path}")

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
        messagebox.showinfo("Saved", f"Preset '{name}' saved.")

    def _load_presets_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            try:
                loaded = load_presets(path)
                self._custom_presets.extend(loaded)
                messagebox.showinfo("Loaded", f"Loaded {len(loaded)} presets.")
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

        # Calorie estimate for 1 hour at current power
        cal_1hr = estimate_calories(rider.power_watts, 3600)
        self.lbl_calories.configure(text=f"~{cal_1hr:.0f} kcal/hr")

        # Power zone
        ftp = self.var_ftp.get()
        zone_name, zone_color = zone_for_power(rider.power_watts, ftp)
        self.lbl_zone.configure(text=zone_name)

        # W/kg on main tab
        w_kg = power_to_weight_ratio(ftp, rider.weight_kg)
        cat = classify_rider(w_kg)
        self.lbl_wkg.configure(text=f"{w_kg:.2f} W/kg ({cat})")

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
        t = self._theme
        ax = self.ax_curve
        ax.clear()
        _style_ax(ax, t)

        powers, speeds = speed_vs_power_curve(rider, bike, course)
        speed_label = "Speed (km/h)" if not self._imperial else "Speed (mph)"
        if self._imperial:
            speeds = speeds * 0.621371

        # Draw power zone bands
        ftp = self.var_ftp.get()
        zones = power_zones(ftp)
        max_speed = max(speeds) if len(speeds) else 50
        for name, lo, hi, color in zones:
            if hi > 3000:
                hi = powers[-1] if len(powers) else 500
            if lo <= powers[-1] if len(powers) else 500:
                ax.axvspan(lo, min(hi, powers[-1] if len(powers) else 500),
                          alpha=0.08, color=color)

        ax.plot(powers, speeds, color=t["ACCENT"], linewidth=2, label="Speed")
        current_speed = result.speed_mph if self._imperial else result.speed_kmh
        ax.axvline(rider.power_watts, color=t["ACCENT4"], linestyle="--", linewidth=1,
                   alpha=0.7, label=f"{rider.power_watts:.0f} W")
        ax.axhline(current_speed, color=t["ACCENT2"], linestyle=":", linewidth=1, alpha=0.5)
        ax.scatter([rider.power_watts], [current_speed], color=t["ACCENT2"], s=60, zorder=5)

        ax.set_xlabel("Power (W)", color=t["FG"], fontsize=9)
        ax.set_ylabel(speed_label, color=t["FG"], fontsize=9)
        ax.set_title("Speed vs Power", color=t["FG"], fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right",
                  facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"], labelcolor=t["FG"])

        ax2 = self.ax_pie
        ax2.clear()
        ax2.set_facecolor(t["BG"])

        labels = []
        sizes = []
        colors = []
        chart_colors = [t[k] for k in CHART_COLOURS_KEYS]
        breakdown = [
            ("Aero", abs(result.power_aero), chart_colors[0]),
            ("Rolling", abs(result.power_rolling), chart_colors[1]),
            ("Gravity", abs(result.power_gravity), chart_colors[2]),
            ("Drivetrain", abs(result.power_drivetrain_loss), chart_colors[3]),
        ]
        for lbl, val, col in breakdown:
            if val > 0.1:
                labels.append(f"{lbl}\n{val:.0f} W")
                sizes.append(val)
                colors.append(col)

        if sizes:
            wedges, texts, autotexts = ax2.pie(
                sizes, labels=labels, colors=colors, autopct="%1.0f%%",
                startangle=90, textprops={"color": t["FG"], "fontsize": 8},
                pctdistance=0.75,
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_color(t["BG"])
                at.set_fontweight("bold")
        ax2.set_title("Power Breakdown", color=t["FG"], fontsize=11, fontweight="bold")

        self.canvas.draw_idle()
