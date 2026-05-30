"""Bike Power Simulator — Tkinter GUI.

Provides an interactive tabbed interface with simulation, course profiles,
workout intervals, GPX import, scenario comparison, real-time ride simulation,
and export capabilities.
"""

from __future__ import annotations

import datetime
import math
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
    PRESETS,
    Preset,
    RealTimeState,
    RiderParams,
    RidingPosition,
    SimulationResult,
    TireType,
    WhatIfChange,
    analyze_ride,
    classify_rider,
    compute_elevation_at_distance,
    compute_realtime_tick,
    estimate_calories,
    estimate_cp_from_ftp,
    export_course_csv,
    export_result_csv,
    load_presets,
    optimize_pacing,
    parse_gpx,
    parse_gpx_ride,
    parse_gpx_with_coords,
    power_to_weight_ratio,
    power_zones,
    predict_kom,
    # Race Planner backend
    MCVariable,
    OptimizeVariable,
    PLANNER_PARAMS,
    monte_carlo_simulation,
    optimize_setup,
    planner_base_value,
    sweep_1d,
    sweep_2d,
    save_presets,
    simulate_course_profile,
    solve_speed,
    speed_vs_power_curve,
    validate_all,
    what_if_analysis,
    zone_for_power,
)

# ---------------------------------------------------------------------------
# Colour palettes — Catppuccin with extended surface layers
# ---------------------------------------------------------------------------
DARK = {
    "BG": "#1e1e2e", "BG_LIGHT": "#262637", "FG": "#cdd6f4",
    "FG_DIM": "#6c7086", "FG_BRIGHT": "#f5f5ff",
    "ACCENT": "#89b4fa", "ACCENT2": "#a6e3a1", "ACCENT3": "#f9e2af",
    "ACCENT4": "#f38ba8", "BORDER": "#393950",
    "ENTRY_BG": "#313244", "ERROR_FG": "#f38ba8",
    "SURFACE0": "#313244", "SURFACE1": "#45475a", "SURFACE2": "#585b70",
    "CARD_BG": "#2a2a3d", "CARD_BORDER": "#3b3b55",
    "TOOLBAR_BG": "#181825", "STATUS_BG": "#11111b",
    "SUCCESS": "#a6e3a1", "WARNING": "#f9e2af", "DANGER": "#f38ba8",
    "GAUGE_BG": "#313244", "GAUGE_TRACK": "#45475a",
}
LIGHT = {
    "BG": "#eff1f5", "BG_LIGHT": "#e6e9ef", "FG": "#4c4f69",
    "FG_DIM": "#8c8fa1", "FG_BRIGHT": "#1e1e2e",
    "ACCENT": "#1e66f5", "ACCENT2": "#40a02b", "ACCENT3": "#df8e1d",
    "ACCENT4": "#d20f39", "BORDER": "#ccd0da",
    "ENTRY_BG": "#ccd0da", "ERROR_FG": "#d20f39",
    "SURFACE0": "#ccd0da", "SURFACE1": "#bcc0cc", "SURFACE2": "#acb0be",
    "CARD_BG": "#dce0e8", "CARD_BORDER": "#bcc0cc",
    "TOOLBAR_BG": "#dce0e8", "STATUS_BG": "#ccd0da",
    "SUCCESS": "#40a02b", "WARNING": "#df8e1d", "DANGER": "#d20f39",
    "GAUGE_BG": "#ccd0da", "GAUGE_TRACK": "#bcc0cc",
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
    ax.tick_params(colors=theme["FG"], labelsize=8, length=3, width=0.5)
    ax.grid(True, alpha=0.15, color=theme["FG"], linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(theme["BORDER"])
        spine.set_linewidth(0.5)


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _convert_speed(speed_kmh: float, imperial: bool) -> float:
    """Convert speed from km/h to mph if imperial."""
    return speed_kmh * 0.621371 if imperial else speed_kmh


def _speed_unit(imperial: bool) -> str:
    return "mph" if imperial else "km/h"


def _dist_unit(imperial: bool) -> str:
    return "mi" if imperial else "km"


def _convert_dist_km(dist_km: float, imperial: bool) -> float:
    """Convert distance from km to miles if imperial."""
    return dist_km * 0.621371 if imperial else dist_km


def _elev_unit(imperial: bool) -> str:
    return "ft" if imperial else "m"


def _convert_elev(elev_m: float, imperial: bool) -> float:
    """Convert elevation from meters to feet if imperial."""
    return elev_m * 3.28084 if imperial else elev_m


# --- Race Planner variable unit handling -----------------------------------
# Backend stores everything in metric; the GUI shows/accepts values in the
# active unit system and converts at the boundary.
_PLANNER_LB_PER_KG = 2.2046226218
_PLANNER_MI_PER_KM = 0.621371


def _planner_unit(key: str, imperial: bool) -> str:
    if key in ("weight_kg", "bike_weight_kg"):
        return "lb" if imperial else "kg"
    if key == "headwind_kmh":
        return "mph" if imperial else "km/h"
    if key == "temperature_c":
        return "\u00b0F" if imperial else "\u00b0C"
    if key == "power_watts":
        return "W"
    if key == "wind_direction_deg":
        return "\u00b0"
    if key == "cda":
        return "m\u00b2"
    return ""  # crr is dimensionless


def _planner_to_display(key: str, v_metric: float, imperial: bool,
                        is_delta: bool = False) -> float:
    """Convert a metric planner value to the active display unit."""
    if not imperial:
        return v_metric
    if key in ("weight_kg", "bike_weight_kg"):
        return v_metric * _PLANNER_LB_PER_KG
    if key == "headwind_kmh":
        return v_metric * _PLANNER_MI_PER_KM
    if key == "temperature_c":
        return v_metric * 9.0 / 5.0 + (0.0 if is_delta else 32.0)
    return v_metric


def _planner_to_metric(key: str, v_disp: float, imperial: bool,
                       is_delta: bool = False) -> float:
    """Convert a display-unit planner value back to metric."""
    if not imperial:
        return v_disp
    if key in ("weight_kg", "bike_weight_kg"):
        return v_disp / _PLANNER_LB_PER_KG
    if key == "headwind_kmh":
        return v_disp / _PLANNER_MI_PER_KM
    if key == "temperature_c":
        return (v_disp - (0.0 if is_delta else 32.0)) * 5.0 / 9.0
    return v_disp


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class BikeSimApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("\u26f2  Bike Power Simulator")
        self.minsize(1280, 860)
        self.geometry("1500x950")

        self._theme = DARK
        self._imperial = False
        self._custom_presets: list[Preset] = []
        self._course_segments: list[dict[str, tk.DoubleVar]] = []
        self._animation_id: str | None = None
        self._last_course_result: Any = None
        self._last_result: SimulationResult | None = None

        # Race Planner state
        self._planner_unit_entries: list[dict[str, Any]] = []
        self._last_mc_result: Any = None
        self._last_sweep1d_result: Any = None
        self._last_sweep2d_result: Any = None
        self._last_optimize_result: Any = None

        # Real-time simulation state
        self._rt_running = False
        self._rt_timer_id: str | None = None
        self._rt_elapsed = 0.0
        self._rt_distance = 0.0
        self._rt_w_prime = 0.0
        self._rt_calories = 0.0
        self._rt_speed_history: list[float] = []
        self._rt_power_history: list[float] = []
        self._rt_time_history: list[float] = []
        self._rt_elev_history: list[float] = []
        self._rt_grade_history: list[float] = []

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
        _font_heading = ("Helvetica", 12, "bold")
        _font_title = ("Helvetica", 14, "bold")

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
                        font=_font_bold, padding=(12, 6))
        style.map("TButton",
                  background=[("active", t["ACCENT2"]), ("disabled", t["BORDER"])],
                  foreground=[("active", "#ffffff"), ("disabled", t.get("FG_DIM", t["BORDER"]))])
        style.configure("Secondary.TButton", background=t["SURFACE0"],
                        foreground=t["FG"], font=_font, padding=(10, 5))
        style.map("Secondary.TButton",
                  background=[("active", t["SURFACE1"])])
        style.configure("Success.TButton", background=t["SUCCESS"],
                        foreground="#1e1e2e", font=_font_bold, padding=(12, 6))
        style.map("Success.TButton",
                  background=[("active", t["ACCENT2"])])
        style.configure("Danger.TButton", background=t["DANGER"],
                        foreground="#ffffff", font=_font_bold, padding=(12, 6))
        style.map("Danger.TButton",
                  background=[("active", t["ACCENT4"])])
        style.configure("TCombobox", fieldbackground=t["ENTRY_BG"],
                        background=t["ENTRY_BG"], foreground=t["FG"],
                        arrowcolor=t["FG"], padding=4)
        style.configure("TCheckbutton", background=t["BG"], foreground=t["FG"])
        style.configure("TNotebook", background=t["BG"], borderwidth=0)
        style.configure("TNotebook.Tab", background=t["SURFACE0"],
                        foreground=t["FG"],
                        padding=[16, 6], font=_font_bold)
        style.map("TNotebook.Tab",
                  background=[("selected", t["ACCENT"])],
                  foreground=[("selected", "#ffffff")])
        style.configure("Result.TLabel", font=("Helvetica", 36, "bold"),
                        foreground=t["ACCENT2"], background=t["BG"])
        style.configure("ResultUnit.TLabel", font=("Helvetica", 14),
                        foreground=t.get("FG_DIM", t["FG"]), background=t["BG"])
        style.configure("Preset.TButton", font=_font_sm, padding=(8, 4),
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
                        rowheight=26)
        style.configure("Treeview.Heading", background=t["SURFACE0"],
                        foreground=t["FG"], font=_font_bold)
        style.configure("Heading.TLabel", font=("Helvetica", 13, "bold"),
                        foreground=t["ACCENT"], background=t["BG"])
        style.configure("TSeparator", background=t["BORDER"])

        # Card styles
        style.configure("Card.TFrame", background=t["CARD_BG"])
        style.configure("Card.TLabel", background=t["CARD_BG"], foreground=t["FG"])
        style.configure("Card.TLabelframe", background=t["CARD_BG"],
                        foreground=t["ACCENT"], borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", background=t["CARD_BG"],
                        foreground=t["ACCENT"], font=_font_heading)

        # Toolbar
        style.configure("Toolbar.TFrame", background=t["TOOLBAR_BG"])
        style.configure("Toolbar.TLabel", background=t["TOOLBAR_BG"],
                        foreground=t["FG"])
        style.configure("ToolbarTitle.TLabel", background=t["TOOLBAR_BG"],
                        foreground=t["ACCENT"], font=_font_title)
        style.configure("ToolbarVersion.TLabel", background=t["TOOLBAR_BG"],
                        foreground=t["FG_DIM"], font=_font_sm)
        style.configure("Toolbar.TButton", background=t["SURFACE0"],
                        foreground=t["FG"], font=_font_sm, padding=(8, 4))
        style.map("Toolbar.TButton",
                  background=[("active", t["SURFACE1"])])

        # Status bar
        style.configure("Status.TFrame", background=t["STATUS_BG"])
        style.configure("Status.TLabel", background=t["STATUS_BG"],
                        foreground=t["FG_DIM"], font=_font_sm)
        style.configure("StatusValue.TLabel", background=t["STATUS_BG"],
                        foreground=t["FG"], font=("Helvetica", 10, "bold"))

        # Dashboard metric cards
        style.configure("MetricCard.TFrame", background=t["CARD_BG"])
        style.configure("MetricValue.TLabel", background=t["CARD_BG"],
                        foreground=t["ACCENT"], font=("Helvetica", 22, "bold"))
        style.configure("MetricLabel.TLabel", background=t["CARD_BG"],
                        foreground=t["FG_DIM"], font=_font_sm)
        style.configure("MetricUnit.TLabel", background=t["CARD_BG"],
                        foreground=t["FG_DIM"], font=("Helvetica", 10))

        # Real-time specific
        style.configure("RT.TFrame", background=t["BG"])
        style.configure("RTMetric.TLabel", background=t["CARD_BG"],
                        foreground=t["ACCENT2"], font=("Helvetica", 28, "bold"))
        style.configure("RTLabel.TLabel", background=t["CARD_BG"],
                        foreground=t["FG_DIM"], font=_font_sm)
        style.configure("RTGrade.TLabel", background=t["CARD_BG"],
                        foreground=t["ACCENT3"], font=("Helvetica", 16, "bold"))

        # TProgressbar
        style.configure("W.Horizontal.TProgressbar",
                        background=t["ACCENT2"], troughcolor=t["GAUGE_TRACK"],
                        borderwidth=0, lightcolor=t["ACCENT2"],
                        darkcolor=t["ACCENT2"])
        style.configure("Danger.Horizontal.TProgressbar",
                        background=t["DANGER"], troughcolor=t["GAUGE_TRACK"],
                        borderwidth=0, lightcolor=t["DANGER"],
                        darkcolor=t["DANGER"])

    def _toggle_theme(self) -> None:
        self._theme = LIGHT if self._theme is DARK else DARK
        self._apply_theme()
        self._theme_btn.configure(
            text="\u263e Dark" if self._theme is LIGHT else "\u2600 Light"
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

        # Real-time simulation variables
        self.var_rt_power = tk.DoubleVar(value=200)
        self.var_rt_speed_mult = tk.DoubleVar(value=10.0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Toolbar
        self._build_toolbar()

        # Main notebook
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 0))

        self._build_main_tab()
        self._build_course_tab()
        self._build_3d_tab()
        self._build_pacing_tab()
        self._build_realtime_tab()
        self._build_ride_analysis_tab()
        self._build_whatif_tab()
        self._build_kom_tab()
        self._build_planner_tab()
        self._build_export_tab()

        # Status bar
        self._build_status_bar()

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self, style="Toolbar.TFrame")
        toolbar.pack(fill=tk.X, side=tk.TOP)

        inner = ttk.Frame(toolbar, style="Toolbar.TFrame")
        inner.pack(fill=tk.X, padx=12, pady=6)

        ttk.Label(inner, text="\u26f2  Bike Power Simulator",
                  style="ToolbarTitle.TLabel").pack(side=tk.LEFT)
        ttk.Label(inner, text="  v6.0",
                  style="ToolbarVersion.TLabel").pack(side=tk.LEFT, padx=(4, 0))

        # Right side controls
        right = ttk.Frame(inner, style="Toolbar.TFrame")
        right.pack(side=tk.RIGHT)

        self._theme_btn = ttk.Button(right, text="\u2600 Light",
                                      style="Toolbar.TButton",
                                      command=self._toggle_theme)
        self._theme_btn.pack(side=tk.RIGHT, padx=4)

        self._unit_btn = ttk.Button(right, text="\u21c4 Imperial",
                                     style="Toolbar.TButton",
                                     command=self._toggle_units)
        self._unit_btn.pack(side=tk.RIGHT, padx=4)

        ttk.Checkbutton(right, text="Auto-update",
                        variable=self.var_auto_update,
                        style="TCheckbutton").pack(side=tk.RIGHT, padx=8)

    def _build_status_bar(self) -> None:
        status = ttk.Frame(self, style="Status.TFrame")
        status.pack(fill=tk.X, side=tk.BOTTOM)

        inner = ttk.Frame(status, style="Status.TFrame")
        inner.pack(fill=tk.X, padx=12, pady=4)

        # Speed
        ttk.Label(inner, text="Speed:", style="Status.TLabel").pack(side=tk.LEFT, padx=(0, 2))
        self._status_speed = ttk.Label(inner, text="0.0 km/h", style="StatusValue.TLabel")
        self._status_speed.pack(side=tk.LEFT, padx=(0, 16))

        # Separator
        ttk.Label(inner, text="\u2502", style="Status.TLabel").pack(side=tk.LEFT, padx=4)

        # Power Zone
        ttk.Label(inner, text="Zone:", style="Status.TLabel").pack(side=tk.LEFT, padx=(4, 2))
        self._status_zone = ttk.Label(inner, text="Z2 Endurance", style="StatusValue.TLabel")
        self._status_zone.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(inner, text="\u2502", style="Status.TLabel").pack(side=tk.LEFT, padx=4)

        # W/kg
        ttk.Label(inner, text="W/kg:", style="Status.TLabel").pack(side=tk.LEFT, padx=(4, 2))
        self._status_wkg = ttk.Label(inner, text="0.00", style="StatusValue.TLabel")
        self._status_wkg.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(inner, text="\u2502", style="Status.TLabel").pack(side=tk.LEFT, padx=4)

        # Calories
        ttk.Label(inner, text="Cal/hr:", style="Status.TLabel").pack(side=tk.LEFT, padx=(4, 2))
        self._status_cal = ttk.Label(inner, text="0", style="StatusValue.TLabel")
        self._status_cal.pack(side=tk.LEFT, padx=(0, 16))

        # Right side: timestamp
        self._status_time = ttk.Label(inner, text="", style="Status.TLabel")
        self._status_time.pack(side=tk.RIGHT, padx=4)

    def _update_status_bar(self, result: SimulationResult, rider: RiderParams) -> None:
        speed_text = f"{result.speed_mph:.1f} mph" if self._imperial else f"{result.speed_kmh:.1f} km/h"
        self._status_speed.configure(text=speed_text)

        ftp = self.var_ftp.get()
        zone_name, _ = zone_for_power(rider.power_watts, ftp)
        self._status_zone.configure(text=zone_name)

        w_kg = power_to_weight_ratio(ftp, rider.weight_kg)
        cat = classify_rider(w_kg)
        self._status_wkg.configure(text=f"{w_kg:.2f} ({cat})")

        cal_hr = estimate_calories(rider.power_watts, 3600)
        self._status_cal.configure(text=f"{cal_hr:.0f} kcal")

        now = datetime.datetime.now().strftime("%H:%M:%S")
        self._status_time.configure(text=now)

    # ------------------------------------------------------------------
    # Card frame helper
    # ------------------------------------------------------------------
    def _make_card(self, parent: tk.Widget, title: str = "",
                   padx: int = 8, pady: int = 4) -> ttk.Frame:
        """Create a card-style frame with optional title."""
        if title:
            card = ttk.LabelFrame(parent, text=title, style="Card.TLabelframe")
        else:
            card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill=tk.X, padx=padx, pady=pady)
        return card

    def _make_metric_card(self, parent: tk.Widget, label: str, value: str,
                          unit: str = "") -> dict[str, ttk.Label]:
        """Create a dashboard metric card and return references."""
        card = ttk.Frame(parent, style="MetricCard.TFrame")
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        inner = ttk.Frame(card, style="MetricCard.TFrame")
        inner.pack(padx=12, pady=8)

        lbl = ttk.Label(inner, text=label, style="MetricLabel.TLabel")
        lbl.pack(anchor=tk.W)

        val_frame = ttk.Frame(inner, style="MetricCard.TFrame")
        val_frame.pack(anchor=tk.W)

        val_lbl = ttk.Label(val_frame, text=value, style="MetricValue.TLabel")
        val_lbl.pack(side=tk.LEFT)

        unit_lbl = ttk.Label(val_frame, text=f" {unit}" if unit else "",
                             style="MetricUnit.TLabel")
        unit_lbl.pack(side=tk.LEFT, anchor=tk.S, pady=(0, 2))

        return {"label": lbl, "value": val_lbl, "unit": unit_lbl, "card": card}

    # ==================================================================
    # TAB 1: Main Simulation
    # ==================================================================
    def _build_main_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  \u26a1 Simulation  ")

        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(6, 3))

        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 6))

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
        frame = ttk.LabelFrame(parent, text="\u2b50 Presets", style="Card.TLabelframe")
        frame.pack(fill=tk.X, pady=(0, 6))
        inner = ttk.Frame(frame, style="Card.TFrame")
        inner.pack(fill=tk.X, padx=8, pady=4)

        all_presets = PRESETS + self._custom_presets
        for i, preset in enumerate(all_presets):
            btn = ttk.Button(
                inner, text=preset.name, style="Preset.TButton",
                command=lambda p=preset: self._apply_preset(p),
            )
            btn.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

        btn_row = ttk.Frame(frame, style="Card.TFrame")
        btn_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Button(btn_row, text="Save Preset", style="Preset.TButton",
                   command=self._save_current_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Load Presets", style="Preset.TButton",
                   command=self._load_presets_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Export Presets", style="Preset.TButton",
                   command=self._export_presets_file).pack(side=tk.LEFT, padx=2)

    def _build_rider_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="\U0001f6b4 Rider", style="Card.TLabelframe")
        frame.pack(fill=tk.X, pady=(0, 6))
        self._add_slider(frame, "Power", self.var_power, 0, 1500, 1, "W")
        self._add_slider(frame, "Weight", self.var_rider_weight, 30, 150, 0.5, "kg")
        self._add_slider(frame, "Height", self.var_rider_height, 140, 210, 1, "cm")
        self._add_slider(frame, "FTP", self.var_ftp, 50, 500, 5, "W")

    def _build_bike_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="\U0001f6b2 Bike", style="Card.TLabelframe")
        frame.pack(fill=tk.X, pady=(0, 6))
        self._add_slider(frame, "Bike Weight", self.var_bike_weight, 3, 25, 0.1, "kg")

        row = ttk.Frame(frame, style="Card.TFrame")
        row.pack(fill=tk.X, padx=10, pady=3)
        _label(row, "Tire Type:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_tire,
                     values=[t.value for t in TireType],
                     state="readonly", width=18).pack(side=tk.RIGHT)

        row2 = ttk.Frame(frame, style="Card.TFrame")
        row2.pack(fill=tk.X, padx=10, pady=3)
        _label(row2, "Position:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Combobox(row2, textvariable=self.var_position,
                     values=[p.value for p in RidingPosition],
                     state="readonly", width=18).pack(side=tk.RIGHT)

        self._add_slider(frame, "Drivetrain Eff.", self.var_efficiency, 85, 100, 0.5, "%")
        self._add_slider(frame, "Wheel Mass", self.var_wheel_mass, 0.5, 4.0, 0.1, "kg")

    def _build_course_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="\U0001f3d4 Course / Environment",
                               style="Card.TLabelframe")
        frame.pack(fill=tk.X, pady=(0, 6))
        self._add_slider(frame, "Grade", self.var_grade, -20, 25, 0.1, "%")
        self._add_slider(frame, "Wind Speed", self.var_wind, 0, 80, 1, "km/h")
        self._add_slider(frame, "Wind Dir", self.var_wind_dir, 0, 360, 5, "\u00b0")
        self._add_slider(frame, "Elevation", self.var_elevation, 0, 5000, 10, "m")
        self._add_slider(frame, "Temperature", self.var_temperature, -10, 50, 1, "\u00b0C")

    def _build_controls(self, parent: tk.Widget) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(frame, text="\u25b6 Calculate",
                   command=self._run_simulation).pack(side=tk.RIGHT, padx=4)

    def _build_results(self, parent: tk.Widget) -> None:
        # Dashboard metric cards row
        dashboard = ttk.Frame(parent)
        dashboard.pack(fill=tk.X, pady=(0, 4))

        # Speed display (large)
        speed_card = ttk.Frame(dashboard, style="MetricCard.TFrame")
        speed_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        inner_sp = ttk.Frame(speed_card, style="MetricCard.TFrame")
        inner_sp.pack(padx=12, pady=8)
        ttk.Label(inner_sp, text="SPEED", style="MetricLabel.TLabel").pack(anchor=tk.W)
        sp_row = ttk.Frame(inner_sp, style="MetricCard.TFrame")
        sp_row.pack(anchor=tk.W)
        self.lbl_speed_primary = ttk.Label(sp_row, text="0.0", style="Result.TLabel")
        self.lbl_speed_primary.configure(background=self._theme["CARD_BG"])
        self.lbl_speed_primary.pack(side=tk.LEFT)
        self.lbl_speed_unit1 = ttk.Label(sp_row, text="km/h",
                                         style="ResultUnit.TLabel")
        self.lbl_speed_unit1.configure(background=self._theme["CARD_BG"])
        self.lbl_speed_unit1.pack(side=tk.LEFT, anchor=tk.S, pady=(0, 8))

        sp_row2 = ttk.Frame(inner_sp, style="MetricCard.TFrame")
        sp_row2.pack(anchor=tk.W)
        self.lbl_speed_secondary = ttk.Label(sp_row2, text="0.0",
                                              font=("Helvetica", 16),
                                              foreground=self._theme["FG_DIM"],
                                              background=self._theme["CARD_BG"])
        self.lbl_speed_secondary.pack(side=tk.LEFT)
        self.lbl_speed_unit2 = ttk.Label(sp_row2, text=" mph",
                                          font=("Helvetica", 11),
                                          foreground=self._theme["FG_DIM"],
                                          background=self._theme["CARD_BG"])
        self.lbl_speed_unit2.pack(side=tk.LEFT, anchor=tk.S, pady=(0, 2))

        # Right-side metric cards
        self._metric_zone = self._make_metric_card(dashboard, "POWER ZONE", "Z2", "")
        self._metric_wkg = self._make_metric_card(dashboard, "W/KG", "0.00", "")
        self._metric_cal = self._make_metric_card(dashboard, "CALORIES", "0", "kcal/hr")

        # Details row
        detail_frame = ttk.Frame(parent)
        detail_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.lbl_details = ttk.Label(detail_frame, text="", style="Small.TLabel",
                                     wraplength=700, justify=tk.LEFT)
        self.lbl_details.pack(anchor=tk.W)

    def _build_charts(self, parent: tk.Widget) -> None:
        t = self._theme
        self.fig = Figure(figsize=(7, 4.5), dpi=100, facecolor=t["BG"])
        self.fig.subplots_adjust(hspace=0.50, left=0.10, right=0.95,
                                 top=0.93, bottom=0.10)
        self.ax_curve = self.fig.add_subplot(211)
        self.ax_pie = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ==================================================================
    # TAB 2: Course Profile
    # ==================================================================
    def _build_course_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  \U0001f4c8 Course Profile  ")

        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=4)

        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=4)

        seg_frame = ttk.LabelFrame(left, text="Course Segments", style="Card.TLabelframe")
        seg_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        btn_row = ttk.Frame(seg_frame, style="Card.TFrame")
        btn_row.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(btn_row, text="+ Segment", style="Preset.TButton",
                   command=self._add_course_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="\u2212 Last", style="Preset.TButton",
                   command=self._remove_course_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="\u25b2", style="Preset.TButton",
                   command=self._move_segment_up).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="\u25bc", style="Preset.TButton",
                   command=self._move_segment_down).pack(side=tk.LEFT, padx=2)

        btn_row2 = ttk.Frame(seg_frame, style="Card.TFrame")
        btn_row2.pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(btn_row2, text="Import GPX", style="Preset.TButton",
                   command=self._import_gpx).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row2, text="\u25b6 Run Profile",
                   command=self._run_course_profile).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_row2, text="\u25b6 Animate", style="Preset.TButton",
                   command=self._animate_course).pack(side=tk.RIGHT, padx=2)

        self._seg_listbox = tk.Listbox(seg_frame, bg=self._theme["ENTRY_BG"],
                                        fg=self._theme["FG"], height=12,
                                        selectmode=tk.SINGLE, font=("Helvetica", 9),
                                        relief="flat", bd=0,
                                        highlightthickness=1,
                                        highlightcolor=self._theme["ACCENT"])
        self._seg_listbox.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

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

    def _add_course_segment(self, distance: float = 1000,
                            grade: float = 0, wind: float = 0,
                            wind_dir: float = 0, elevation: float = 100,
                            temperature: float = 20) -> None:
        seg_vars: dict[str, tk.DoubleVar] = {
            "distance": tk.DoubleVar(value=distance),
            "grade": tk.DoubleVar(value=grade),
            "wind": tk.DoubleVar(value=wind),
            "wind_dir": tk.DoubleVar(value=wind_dir),
            "elevation": tk.DoubleVar(value=elevation),
            "temperature": tk.DoubleVar(value=temperature),
        }
        self._course_segments.append(seg_vars)
        self._refresh_seg_listbox()

    def _remove_course_segment(self) -> None:
        if self._course_segments:
            self._course_segments.pop()
            self._refresh_seg_listbox()

    def _refresh_seg_listbox(self) -> None:
        self._seg_listbox.delete(0, tk.END)
        for i, sv in enumerate(self._course_segments):
            d = sv["distance"].get()
            g = sv["grade"].get()
            self._seg_listbox.insert(
                tk.END,
                f"  Seg {i+1}: {d:.0f}m  |  {g:+.1f}%  |  "
                f"wind {sv['wind'].get():.0f} km/h"
            )

    def _move_segment_up(self) -> None:
        sel = self._seg_listbox.curselection()
        if sel and sel[0] > 0:
            idx = sel[0]
            self._course_segments[idx - 1], self._course_segments[idx] = (
                self._course_segments[idx], self._course_segments[idx - 1]
            )
            self._refresh_seg_listbox()
            self._seg_listbox.selection_set(idx - 1)

    def _move_segment_down(self) -> None:
        sel = self._seg_listbox.curselection()
        if sel and sel[0] < len(self._course_segments) - 1:
            idx = sel[0]
            self._course_segments[idx], self._course_segments[idx + 1] = (
                self._course_segments[idx + 1], self._course_segments[idx]
            )
            self._refresh_seg_listbox()
            self._seg_listbox.selection_set(idx + 1)

    def _import_gpx(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("GPX files", "*.gpx")])
        if not path:
            return
        try:
            segments = parse_gpx(path)
            if not segments:
                messagebox.showwarning("GPX", "No segments found in the file.")
                return
            self._course_segments.clear()
            for seg in segments:
                self._add_course_segment(
                    distance=seg.distance_m, grade=seg.grade_pct,
                    wind=seg.headwind_kmh, wind_dir=seg.wind_direction_deg,
                    elevation=seg.elevation_m, temperature=seg.temperature_c,
                )
            messagebox.showinfo("GPX Loaded",
                                f"Loaded {len(segments)} segments from GPX.")
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
                pass

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
        imp = self._imperial
        su = _speed_unit(imp)
        du = _dist_unit(imp)
        eu = _elev_unit(imp)
        dist_val = _convert_dist_km(result.total_distance_m / 1000, imp)
        avg_spd = _convert_speed(result.avg_speed_kmh, imp)
        elev_gain = _convert_elev(result.total_elevation_gain_m, imp)
        elev_loss = _convert_elev(result.total_elevation_loss_m, imp)
        summary = (
            f"Total: {dist_val:.2f} {du}  \u2502  "
            f"Time: {mins}:{secs:02d}\n"
            f"Avg Speed: {avg_spd:.1f} {su}\n"
            f"Elev Gain: {elev_gain:.0f} {eu}  \u2502  "
            f"Loss: {elev_loss:.0f} {eu}\n"
            f"Est. Calories: {total_cal:.0f} kcal"
        )
        self.lbl_course_summary.configure(text=summary)
        self._update_course_charts(result)

    def _update_course_charts(self, result: Any) -> None:
        t = self._theme
        imp = self._imperial
        su = _speed_unit(imp)
        du = _dist_unit(imp)
        eu = _elev_unit(imp)

        distances_km = [_convert_dist_km(d / 1000, imp) for d in result.distances_cumulative]
        speeds = [_convert_speed(s, imp) for s in result.speeds]
        avg_speed = _convert_speed(result.avg_speed_kmh, imp)
        elevations = [_convert_elev(e, imp) for e in result.elevations]

        ax1 = self.ax_course_speed
        ax1.clear()
        _style_ax(ax1, t)
        mid_distances = [(distances_km[i] + distances_km[i + 1]) / 2
                         for i in range(len(speeds))]
        ax1.bar(mid_distances, speeds, width=[
            distances_km[i + 1] - distances_km[i] for i in range(len(speeds))
        ], color=t["ACCENT"], alpha=0.8, edgecolor=t["BORDER"], linewidth=0.5)
        ax1.axhline(avg_speed, color=t["ACCENT4"], linestyle="--",
                    linewidth=1, label=f"Avg: {avg_speed:.1f} {su}")
        ax1.set_xlabel(f"Distance ({du})", color=t["FG"], fontsize=9)
        ax1.set_ylabel(f"Speed ({su})", color=t["FG"], fontsize=9)
        ax1.set_title("Speed by Segment", color=t["FG"], fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"])

        ax2 = self.ax_course_elev
        ax2.clear()
        _style_ax(ax2, t)
        ax2.fill_between(distances_km, elevations, alpha=0.3, color=t["ACCENT2"])
        ax2.plot(distances_km, elevations, color=t["ACCENT2"], linewidth=2)
        ax2.set_xlabel(f"Distance ({du})", color=t["FG"], fontsize=9)
        ax2.set_ylabel(f"Elevation ({eu})", color=t["FG"], fontsize=9)
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
    # TAB 7: 3D Course Visualization & Gradient Map
    # ==================================================================
    def _build_3d_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  \U0001f30d 3D / Map  ")

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
        imp = self._imperial
        du = _dist_unit(imp)
        eu = _elev_unit(imp)

        raw_distances_km = [d / 1000 for d in result.distances_cumulative]
        raw_elevations = result.elevations
        distances_disp = [_convert_dist_km(d, imp) for d in raw_distances_km]
        elevations_disp = [_convert_elev(e, imp) for e in raw_elevations]

        ax = self.ax_3d
        ax.clear()

        ax.set_facecolor(t["BG_LIGHT"])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        n = len(distances_disp)
        x = distances_disp
        y = [0.0] * n
        z = elevations_disp

        for i in range(n - 1):
            grade = 0.0
            dist_m = (raw_distances_km[i + 1] - raw_distances_km[i]) * 1000
            if dist_m > 0:
                grade = (raw_elevations[i + 1] - raw_elevations[i]) / dist_m * 100

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

        ax.plot(x, [0] * n, [min(z)] * n, linestyle=":", linewidth=0.5,
                    color=t["ACCENT2"], alpha=0.15)

        ax.set_xlabel(f"Distance ({du})", color=t["FG"], fontsize=8, labelpad=8)
        ax.set_ylabel("", color=t["FG"], fontsize=8)
        ax.set_zlabel(f"Elevation ({eu})", color=t["FG"], fontsize=8, labelpad=8)
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
        imp = self._imperial
        du = _dist_unit(imp)
        eu = _elev_unit(imp)

        distances_km = [_convert_dist_km(d / 1000, imp) for d in result.distances_cumulative]
        elevations = [_convert_elev(e, imp) for e in result.elevations]

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

        ax.set_xlabel(f"Distance ({du})", color=t["FG"], fontsize=8, labelpad=8)
        ax.set_ylabel("", color=t["FG"], fontsize=8)
        ax.set_zlabel(f"Elevation ({eu})", color=t["FG"], fontsize=8, labelpad=8)
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
    # TAB 9: Route Optimization (Pacing Strategy)
    # ==================================================================
    def _build_pacing_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  \U0001f3af Pacing  ")

        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=4)

        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=4)

        ctrl = ttk.LabelFrame(left, text="Pacing Settings", style="Card.TLabelframe")
        ctrl.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(ctrl, text="Uses course segments from Course Profile tab",
                  style="Small.TLabel").pack(padx=8, pady=2)

        self.var_target_power = tk.DoubleVar(value=200)
        r1 = ttk.Frame(ctrl, style="Card.TFrame")
        r1.pack(fill=tk.X, padx=8, pady=2)
        _label(r1, "Target Avg Power (W):", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.var_target_power, width=6).pack(side=tk.RIGHT)

        ttk.Button(ctrl, text="\u25b6 Optimize Pacing",
                   command=self._run_pacing).pack(fill=tk.X, padx=8, pady=4)

        res_frame = ttk.LabelFrame(left, text="Optimization Results",
                                    style="Card.TLabelframe")
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
            messagebox.showwarning("No Segments",
                                   "Add course segments in the Course Profile tab first.")
            return

        target_power = self.var_target_power.get()
        result = optimize_pacing(rider, bike, segments, ftp,
                                  target_avg_power=target_power)

        mins = int(result.total_time_s // 60)
        secs = int(result.total_time_s % 60)
        even_mins = int(result.even_pace_time_s // 60)
        even_secs = int(result.even_pace_time_s % 60)

        imp = self._imperial
        su = _speed_unit(imp)
        avg_spd = _convert_speed(result.avg_speed_kmh, imp)
        text = (
            f"Optimized Time: {mins}:{secs:02d}\n"
            f"Even-Pace Time: {even_mins}:{even_secs:02d}\n"
            f"Time Saved: {result.time_saved_s:.1f}s\n"
            f"Avg Power: {result.avg_power:.0f} W\n"
            f"Avg Speed: {avg_spd:.1f} {su}\n\n"
            f"Segment Strategy:\n"
        )
        for ps in result.segments:
            seg_spd = _convert_speed(ps.speed_kmh, imp)
            text += (
                f"  Seg {ps.segment_index+1}: {ps.grade_pct:+.1f}% "
                f"\u2192 {ps.optimal_power:.0f}W "
                f"({seg_spd:.1f} {su}) "
                f"[{ps.strategy_note}]\n"
            )

        self.lbl_pacing_result.configure(text=text.strip())
        self._update_pacing_charts(result)

    def _update_pacing_charts(self, result: Any) -> None:
        t = self._theme
        imp = self._imperial
        su = _speed_unit(imp)

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

        avg_spd = _convert_speed(result.avg_speed_kmh, imp)
        speeds = [_convert_speed(ps.speed_kmh, imp) for ps in result.segments]
        ax2.bar(seg_indices, speeds, color=t["ACCENT"], edgecolor=t["BORDER"],
                alpha=0.85)
        ax2.axhline(avg_spd, color=t["ACCENT4"], linestyle="--",
                    linewidth=1.5, label=f"Avg: {avg_spd:.1f} {su}")
        for i, (idx, s) in enumerate(zip(seg_indices, speeds)):
            ax2.text(idx, s + 0.5, f"{s:.1f}",
                     ha="center", va="bottom", color=t["FG"], fontsize=7)

        ax2.set_xlabel("Segment", color=t["FG"], fontsize=9)
        ax2.set_ylabel(f"Speed ({su})", color=t["FG"], fontsize=9)
        ax2.set_title("Resulting Speed by Segment", color=t["FG"],
                      fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"])

        self.canvas_pacing.draw_idle()

    # ==================================================================
    # TAB 10: Real-Time Live Ride Simulation
    # ==================================================================
    def _build_realtime_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  \u23f1 Live Ride  ")

        # Left panel: controls + live metrics
        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=4)

        # Controls card
        ctrl = ttk.LabelFrame(left, text="\u25b6 Ride Controls",
                               style="Card.TLabelframe")
        ctrl.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(ctrl, text="Uses course from Course Profile tab",
                  style="Small.TLabel").pack(padx=8, pady=(4, 2))

        btn_row = ttk.Frame(ctrl, style="Card.TFrame")
        btn_row.pack(fill=tk.X, padx=8, pady=4)

        self._rt_start_btn = ttk.Button(btn_row, text="\u25b6 Start",
                                         style="Success.TButton",
                                         command=self._rt_start)
        self._rt_start_btn.pack(side=tk.LEFT, padx=2)

        self._rt_pause_btn = ttk.Button(btn_row, text="\u23f8 Pause",
                                         style="Secondary.TButton",
                                         command=self._rt_pause)
        self._rt_pause_btn.pack(side=tk.LEFT, padx=2)

        self._rt_reset_btn = ttk.Button(btn_row, text="\u21ba Reset",
                                         style="Danger.TButton",
                                         command=self._rt_reset)
        self._rt_reset_btn.pack(side=tk.LEFT, padx=2)

        # Speed multiplier
        speed_frame = ttk.Frame(ctrl, style="Card.TFrame")
        speed_frame.pack(fill=tk.X, padx=8, pady=2)
        _label(speed_frame, "Sim Speed:", style="Card.TLabel").pack(side=tk.LEFT)
        self._rt_speed_lbl = _label(speed_frame, "10x", style="Card.TLabel")
        self._rt_speed_lbl.pack(side=tk.RIGHT, padx=(4, 0))
        speed_scale = _make_scale(speed_frame, 1, 50, self.var_rt_speed_mult,
                                  command=self._rt_speed_changed)
        speed_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=4)

        # Live power slider
        power_frame = ttk.Frame(ctrl, style="Card.TFrame")
        power_frame.pack(fill=tk.X, padx=8, pady=(2, 6))
        _label(power_frame, "Live Power:", style="Card.TLabel").pack(side=tk.LEFT)
        self._rt_power_lbl = _label(power_frame, "200 W", style="Card.TLabel")
        self._rt_power_lbl.pack(side=tk.RIGHT, padx=(4, 0))
        power_scale = _make_scale(power_frame, 0, 600, self.var_rt_power,
                                  command=self._rt_power_changed)
        power_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=4)

        # Live metrics dashboard
        metrics = ttk.LabelFrame(left, text="\U0001f4ca Live Metrics",
                                  style="Card.TLabelframe")
        metrics.pack(fill=tk.X, pady=(0, 6))

        # Speed display
        speed_card = ttk.Frame(metrics, style="MetricCard.TFrame")
        speed_card.pack(fill=tk.X, padx=8, pady=4)
        self._rt_speed_display = ttk.Label(speed_card, text="0.0",
                                            style="RTMetric.TLabel")
        self._rt_speed_display.pack(side=tk.LEFT, padx=8)
        self._rt_speed_unit_lbl = ttk.Label(speed_card, text="km/h", style="RTLabel.TLabel")
        self._rt_speed_unit_lbl.pack(side=tk.LEFT, anchor=tk.S, pady=(0, 6))
        self._rt_grade_display = ttk.Label(speed_card, text="0.0%",
                                            style="RTGrade.TLabel")
        self._rt_grade_display.pack(side=tk.RIGHT, padx=8)
        ttk.Label(speed_card, text="grade", style="RTLabel.TLabel").pack(
            side=tk.RIGHT, anchor=tk.S, pady=(0, 4))

        # Metrics grid
        grid_frame = ttk.Frame(metrics, style="Card.TFrame")
        grid_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        self._rt_labels: dict[str, ttk.Label] = {}
        metric_defs = [
            ("Distance", "0.00 km", 0, 0),
            ("Elapsed", "0:00", 0, 1),
            ("Avg Speed", "0.0", 1, 0),
            ("Calories", "0 kcal", 1, 1),
            ("Segment", "1 / 1", 2, 0),
            ("Zone", "Z2", 2, 1),
        ]
        for name, default, row, col in metric_defs:
            cell = ttk.Frame(grid_frame, style="Card.TFrame")
            cell.grid(row=row, column=col, sticky="ew", padx=4, pady=2)
            ttk.Label(cell, text=name, style="RTLabel.TLabel").pack(anchor=tk.W)
            lbl = ttk.Label(cell, text=default,
                            font=("Helvetica", 12, "bold"),
                            foreground=self._theme["FG"],
                            background=self._theme["CARD_BG"])
            lbl.pack(anchor=tk.W)
            self._rt_labels[name] = lbl
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

        # W' Balance bar
        wprime_frame = ttk.LabelFrame(left, text="W' Balance",
                                       style="Card.TLabelframe")
        wprime_frame.pack(fill=tk.X, pady=(0, 6))

        self._rt_wprime_pct = ttk.Label(wprime_frame, text="100%",
                                         font=("Helvetica", 14, "bold"),
                                         foreground=self._theme["ACCENT2"],
                                         background=self._theme["CARD_BG"])
        self._rt_wprime_pct.pack(padx=8, anchor=tk.W, pady=(4, 0))

        self._rt_wprime_bar = ttk.Progressbar(
            wprime_frame, mode="determinate", maximum=100,
            style="W.Horizontal.TProgressbar", length=250)
        self._rt_wprime_bar.pack(fill=tk.X, padx=8, pady=(2, 6))
        self._rt_wprime_bar["value"] = 100

        self._rt_wprime_detail = ttk.Label(wprime_frame, text="20.0 / 20.0 kJ",
                                            style="Small.TLabel")
        self._rt_wprime_detail.pack(padx=8, anchor=tk.W, pady=(0, 4))

        # Right panel: charts
        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=4)

        t = self._theme
        self.fig_rt = Figure(figsize=(8, 6), dpi=100, facecolor=t["BG"])
        self.fig_rt.subplots_adjust(hspace=0.45, left=0.10, right=0.95,
                                     top=0.95, bottom=0.08)
        self.ax_rt_course = self.fig_rt.add_subplot(311)
        self.ax_rt_speed = self.fig_rt.add_subplot(312)
        self.ax_rt_power = self.fig_rt.add_subplot(313)
        self.canvas_rt = FigureCanvasTkAgg(self.fig_rt, master=right)
        self.canvas_rt.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _rt_speed_changed(self, value: str) -> None:
        v = int(float(value))
        self.var_rt_speed_mult.set(v)
        self._rt_speed_lbl.configure(text=f"{v}x")

    def _rt_power_changed(self, value: str) -> None:
        v = int(float(value))
        self.var_rt_power.set(v)
        self._rt_power_lbl.configure(text=f"{v} W")

    def _rt_get_segments(self) -> list[CourseSegment]:
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
                CourseSegment(distance_m=2000, grade_pct=0),
                CourseSegment(distance_m=1500, grade_pct=5),
                CourseSegment(distance_m=1000, grade_pct=-3),
                CourseSegment(distance_m=2000, grade_pct=0),
            ]
        return segments

    def _rt_draw_base_course(self, segments: list[CourseSegment]) -> None:
        """Draw the full course elevation profile as background."""
        t = self._theme
        imp = self._imperial
        du = _dist_unit(imp)
        eu = _elev_unit(imp)
        ax = self.ax_rt_course
        ax.clear()
        _style_ax(ax, t)

        total_dist = sum(s.distance_m for s in segments)
        n_points = max(100, int(total_dist / 10))
        distances = [i * total_dist / n_points for i in range(n_points + 1)]
        elevations_raw = [compute_elevation_at_distance(segments, d) for d in distances]
        distances_disp = [_convert_dist_km(d / 1000, imp) for d in distances]
        elevations_disp = [_convert_elev(e, imp) for e in elevations_raw]

        ax.fill_between(distances_disp, elevations_disp, alpha=0.2, color=t["ACCENT2"])
        ax.plot(distances_disp, elevations_disp, color=t["ACCENT2"], linewidth=1.5, alpha=0.5)

        # Grade color coding on segments
        cum = 0.0
        for seg in segments:
            seg_start = _convert_dist_km(cum / 1000, imp)
            seg_end = _convert_dist_km((cum + seg.distance_m) / 1000, imp)
            if seg.grade_pct > 5:
                color = t["ACCENT4"]
            elif seg.grade_pct > 2:
                color = t["ACCENT3"]
            elif seg.grade_pct > 0:
                color = t["ACCENT2"]
            else:
                color = t["ACCENT"]
            ax.axvspan(seg_start, seg_end, alpha=0.05, color=color)
            cum += seg.distance_m

        ax.set_ylabel(f"Elevation ({eu})", color=t["FG"], fontsize=9)
        ax.set_title("Course Profile — Live Ride", color=t["FG"],
                     fontsize=11, fontweight="bold")

        # Rider dot (will be updated)
        self._rt_dot, = ax.plot([], [], "o", color=t["ACCENT4"], markersize=12,
                                 zorder=10, markeredgecolor="#ffffff",
                                 markeredgewidth=2)

        self.canvas_rt.draw_idle()

    def _rt_start(self) -> None:
        if self._rt_running:
            return

        segments = self._rt_get_segments()
        if not segments:
            messagebox.showwarning("No Course", "Add course segments first.")
            return

        if self._rt_elapsed == 0.0:
            # Fresh start
            self._rt_segments = segments
            self._rt_distance = 0.0
            self._rt_elapsed = 0.0
            self._rt_calories = 0.0
            ftp = self.var_ftp.get()
            self._rt_cp_model = estimate_cp_from_ftp(ftp)
            self._rt_w_prime = self._rt_cp_model.w_prime_joules
            self._rt_speed_history = []
            self._rt_power_history = []
            self._rt_time_history = []
            self._rt_elev_history = []
            self._rt_grade_history = []
            self._rt_draw_base_course(segments)

        self._rt_running = True
        self._rt_start_btn.configure(state="disabled")
        self._rt_tick()

    def _rt_pause(self) -> None:
        self._rt_running = False
        self._rt_start_btn.configure(state="normal")
        if self._rt_timer_id:
            self.after_cancel(self._rt_timer_id)
            self._rt_timer_id = None

    def _rt_reset(self) -> None:
        self._rt_pause()
        self._rt_elapsed = 0.0
        self._rt_distance = 0.0
        self._rt_calories = 0.0
        self._rt_w_prime = 0.0
        self._rt_speed_history = []
        self._rt_power_history = []
        self._rt_time_history = []
        self._rt_elev_history = []
        self._rt_grade_history = []

        self._rt_speed_display.configure(text="0.0")
        self._rt_grade_display.configure(text="0.0%")
        self._rt_wprime_bar["value"] = 100
        self._rt_wprime_pct.configure(text="100%")
        for name in self._rt_labels:
            defaults = {"Distance": "0.00 km", "Elapsed": "0:00",
                        "Avg Speed": "0.0", "Calories": "0 kcal",
                        "Segment": "1 / 1", "Zone": "Z2"}
            self._rt_labels[name].configure(text=defaults.get(name, ""))

        # Clear charts
        for ax in [self.ax_rt_course, self.ax_rt_speed, self.ax_rt_power]:
            ax.clear()
        self.canvas_rt.draw_idle()

    def _rt_tick(self) -> None:
        if not self._rt_running:
            return

        rider, bike, _ = self._gather_params()
        power = self.var_rt_power.get()
        ftp = self.var_ftp.get()
        speed_mult = self.var_rt_speed_mult.get()
        dt = 1.0 * speed_mult  # simulated seconds per tick

        state = compute_realtime_tick(
            rider, bike, self._rt_segments, power, ftp,
            self._rt_cp_model,
            self._rt_elapsed, dt,
            self._rt_distance, self._rt_w_prime, self._rt_calories,
        )

        self._rt_elapsed = state.elapsed_time_s
        self._rt_distance = state.distance_m
        self._rt_w_prime = state.w_prime_balance_j
        self._rt_calories = state.calories_kcal

        # Record history
        self._rt_time_history.append(state.elapsed_time_s)
        self._rt_speed_history.append(state.current_speed_kmh)
        self._rt_power_history.append(state.current_power_watts)
        self._rt_elev_history.append(state.current_elevation_m)
        self._rt_grade_history.append(state.current_grade_pct)

        # Update UI
        self._rt_update_display(state)

        if state.is_finished:
            self._rt_running = False
            self._rt_start_btn.configure(state="normal")
            messagebox.showinfo("Ride Complete",
                                f"Finished in {_format_time(state.elapsed_time_s)}\n"
                                f"Distance: {_convert_dist_km(state.distance_m/1000, self._imperial):.2f} {_dist_unit(self._imperial)}\n"
                                f"Avg Speed: {_convert_speed(state.avg_speed_kmh, self._imperial):.1f} {_speed_unit(self._imperial)}\n"
                                f"Calories: {state.calories_kcal:.0f} kcal")
            return

        self._rt_timer_id = self.after(100, self._rt_tick)

    def _rt_update_display(self, state: RealTimeState) -> None:
        t = self._theme
        imp = self._imperial
        su = _speed_unit(imp)
        du = _dist_unit(imp)

        # Speed
        disp_speed = _convert_speed(state.current_speed_kmh, imp)
        self._rt_speed_display.configure(text=f"{disp_speed:.1f}")
        self._rt_speed_unit_lbl.configure(text=su)

        # Grade
        grade_text = f"{state.current_grade_pct:+.1f}%"
        if state.current_grade_pct > 5:
            grade_color = t["ACCENT4"]
        elif state.current_grade_pct > 2:
            grade_color = t["ACCENT3"]
        elif state.current_grade_pct > 0:
            grade_color = t["ACCENT2"]
        else:
            grade_color = t["ACCENT"]
        self._rt_grade_display.configure(text=grade_text, foreground=grade_color)

        # Metrics
        dist_val = _convert_dist_km(state.distance_m / 1000, imp)
        self._rt_labels["Distance"].configure(
            text=f"{dist_val:.2f} {du}")
        self._rt_labels["Elapsed"].configure(
            text=_format_time(state.elapsed_time_s))
        avg_spd = _convert_speed(state.avg_speed_kmh, imp)
        self._rt_labels["Avg Speed"].configure(
            text=f"{avg_spd:.1f} {su}")
        self._rt_labels["Calories"].configure(
            text=f"{state.calories_kcal:.0f} kcal")
        n_segs = len(self._rt_segments) if hasattr(self, '_rt_segments') else 1
        self._rt_labels["Segment"].configure(
            text=f"{state.segment_index + 1} / {n_segs}")
        self._rt_labels["Zone"].configure(
            text=state.power_zone_name)

        # W' balance
        w_pct = (state.w_prime_balance_j / state.w_prime_max_j * 100
                 if state.w_prime_max_j > 0 else 100)
        self._rt_wprime_bar["value"] = w_pct
        self._rt_wprime_pct.configure(text=f"{w_pct:.0f}%")
        if w_pct > 60:
            self._rt_wprime_pct.configure(foreground=t["SUCCESS"])
            self._rt_wprime_bar.configure(style="W.Horizontal.TProgressbar")
        elif w_pct > 30:
            self._rt_wprime_pct.configure(foreground=t["WARNING"])
            self._rt_wprime_bar.configure(style="W.Horizontal.TProgressbar")
        else:
            self._rt_wprime_pct.configure(foreground=t["DANGER"])
            self._rt_wprime_bar.configure(style="Danger.Horizontal.TProgressbar")
        self._rt_wprime_detail.configure(
            text=f"{state.w_prime_balance_j/1000:.1f} / {state.w_prime_max_j/1000:.1f} kJ")

        # Update charts every 5 ticks to reduce redraw overhead
        if len(self._rt_time_history) % 5 == 0 or state.is_finished:
            self._rt_update_charts(state)

    def _rt_update_charts(self, state: RealTimeState) -> None:
        t = self._theme
        imp = self._imperial
        su = _speed_unit(imp)

        # Course profile with rider dot
        ax1 = self.ax_rt_course
        if hasattr(self, '_rt_dot'):
            dist_km = _convert_dist_km(state.distance_m / 1000, imp)
            elev = _convert_elev(state.current_elevation_m, imp)
            self._rt_dot.set_data([dist_km], [elev])

        # Speed trace
        ax2 = self.ax_rt_speed
        ax2.clear()
        _style_ax(ax2, t)
        time_mins = [ts / 60 for ts in self._rt_time_history]
        if time_mins:
            disp_speeds = [_convert_speed(s, imp) for s in self._rt_speed_history]
            ax2.plot(time_mins, disp_speeds, color=t["ACCENT"],
                     linewidth=1.5)
            ax2.fill_between(time_mins, disp_speeds,
                             alpha=0.15, color=t["ACCENT"])
            if len(disp_speeds) > 1:
                avg = sum(disp_speeds) / len(disp_speeds)
                ax2.axhline(avg, color=t["ACCENT4"], linestyle="--",
                            linewidth=1, alpha=0.7)
        ax2.set_ylabel(f"Speed ({su})", color=t["FG"], fontsize=9)
        ax2.set_title("Live Speed", color=t["FG"], fontsize=10, fontweight="bold")

        # Power trace
        ax3 = self.ax_rt_power
        ax3.clear()
        _style_ax(ax3, t)
        if time_mins:
            ftp = self.var_ftp.get()
            zones = power_zones(ftp)
            max_pw = max(self._rt_power_history) * 1.1 if self._rt_power_history else 500
            for name, lo, hi, color in zones:
                if hi > 3000:
                    hi = max_pw
                ax3.axhspan(lo, min(hi, max_pw), alpha=0.08, color=color)

            for i in range(len(time_mins) - 1):
                _, color = zone_for_power(self._rt_power_history[i], ftp)
                ax3.fill_between(
                    [time_mins[i], time_mins[i + 1]],
                    [self._rt_power_history[i], self._rt_power_history[i + 1]],
                    alpha=0.5, color=color
                )
            ax3.plot(time_mins, self._rt_power_history, color=t["FG"],
                     linewidth=0.5, alpha=0.5)
        ax3.set_xlabel("Time (min)", color=t["FG"], fontsize=9)
        ax3.set_ylabel("Power (W)", color=t["FG"], fontsize=9)
        ax3.set_title("Live Power", color=t["FG"], fontsize=10, fontweight="bold")

        self.canvas_rt.draw_idle()

    # ==================================================================
    # TAB 11: Ride Analysis
    # ==================================================================
    def _build_ride_analysis_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  \U0001f4ca Ride Analysis  ")

        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left panel — controls
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        ctrl = self._make_card(left, "GPX Ride Import")

        ttk.Button(ctrl, text="Load GPX Ride File",
                   command=self._load_ride_gpx).pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(ctrl, text="Power (W) if not in file:",
                  font=("Helvetica", 9)).pack(anchor=tk.W, padx=8, pady=(8, 0))
        self.var_ride_power = tk.DoubleVar(value=200)
        self._add_slider(ctrl, "Ride Power", self.var_ride_power, 50, 500, 5, "W")

        ttk.Label(ctrl, text="Smoothing Window:",
                  font=("Helvetica", 9)).pack(anchor=tk.W, padx=8, pady=(8, 0))
        self.var_smoothing = tk.IntVar(value=5)
        smooth_frame = ttk.Frame(ctrl)
        smooth_frame.pack(fill=tk.X, padx=8, pady=2)
        for v in [3, 5, 10, 20]:
            ttk.Radiobutton(smooth_frame, text=str(v), variable=self.var_smoothing,
                            value=v).pack(side=tk.LEFT, padx=4)

        ttk.Button(ctrl, text="Re-analyze", style="Secondary.TButton",
                   command=self._reanalyze_ride).pack(fill=tk.X, padx=8, pady=4)

        # Model accuracy card
        acc_card = self._make_card(left, "Model Accuracy")
        self.lbl_ride_accuracy = ttk.Label(acc_card, text="Load a GPX file to begin",
                                            font=("Helvetica", 10), wraplength=280,
                                            justify=tk.LEFT)
        self.lbl_ride_accuracy.pack(padx=8, pady=8, anchor=tk.W)

        # Ride summary card
        summ_card = self._make_card(left, "Ride Summary")
        self.lbl_ride_summary = ttk.Label(summ_card, text="",
                                           font=("Helvetica", 10), wraplength=280,
                                           justify=tk.LEFT)
        self.lbl_ride_summary.pack(padx=8, pady=8, anchor=tk.W)

        # Right panel — charts
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        t = self._theme
        self.fig_ride = Figure(figsize=(8, 8), dpi=100, facecolor=t["BG"])
        self.fig_ride.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06,
                                      hspace=0.35)
        self.ax_ride_speed = self.fig_ride.add_subplot(311)
        self.ax_ride_elev = self.fig_ride.add_subplot(312)
        self.ax_ride_error = self.fig_ride.add_subplot(313)
        self.canvas_ride = FigureCanvasTkAgg(self.fig_ride, master=right)
        self.canvas_ride.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._ride_points: list = []
        self._ride_analysis = None

    def _load_ride_gpx(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("GPX files", "*.gpx")])
        if not path:
            return
        try:
            self._ride_points = parse_gpx_ride(path)
            if not self._ride_points:
                messagebox.showwarning("GPX", "No trackpoints with timestamps found.")
                return
            self._run_ride_analysis()
        except Exception as e:
            messagebox.showerror("GPX Error", str(e))

    def _reanalyze_ride(self) -> None:
        if not self._ride_points:
            messagebox.showwarning("No Data", "Load a GPX ride file first.")
            return
        self._run_ride_analysis()

    def _run_ride_analysis(self) -> None:
        rider, bike, course = self._gather_params()
        rider.power_watts = self.var_ride_power.get()

        try:
            analysis = analyze_ride(
                self._ride_points, rider, bike,
                smoothing_window=self.var_smoothing.get(),
                temperature_c=course.temperature_c,
                headwind_kmh=course.headwind_kmh,
                wind_direction_deg=course.wind_direction_deg,
            )
        except ValueError as e:
            messagebox.showwarning("Analysis Error", str(e))
            return

        self._ride_analysis = analysis
        imp = self._imperial
        su = _speed_unit(imp)
        du = _dist_unit(imp)
        eu = _elev_unit(imp)

        # Accuracy metrics
        mean_err = _convert_speed(abs(analysis.mean_error_kmh), imp)
        rmse = _convert_speed(analysis.rmse_kmh, imp)
        acc_text = (
            f"Mean Error: {mean_err:.1f} {su}\n"
            f"RMSE: {rmse:.1f} {su}\n"
            f"Mean % Error: {analysis.mean_pct_error:.1f}%\n\n"
            f"Predicted Avg: {_convert_speed(analysis.predicted_avg_speed_kmh, imp):.1f} {su}\n"
            f"Actual Avg: {_convert_speed(analysis.actual_avg_speed_kmh, imp):.1f} {su}"
        )
        self.lbl_ride_accuracy.configure(text=acc_text)

        # Ride summary
        dist_val = _convert_dist_km(analysis.total_distance_m / 1000, imp)
        mins = int(analysis.total_time_s // 60)
        secs = int(analysis.total_time_s % 60)
        elev_range = max(analysis.elevations) - min(analysis.elevations) if analysis.elevations else 0
        summ_text = (
            f"Distance: {dist_val:.1f} {du}\n"
            f"Time: {mins}:{secs:02d}\n"
            f"Elev Range: {_convert_elev(elev_range, imp):.0f} {eu}\n"
            f"Points: {len(analysis.actual_speeds)}"
        )
        self.lbl_ride_summary.configure(text=summ_text)

        self._update_ride_charts(analysis)

    def _update_ride_charts(self, analysis: Any) -> None:
        t = self._theme
        imp = self._imperial
        su = _speed_unit(imp)
        du = _dist_unit(imp)
        eu = _elev_unit(imp)

        distances = [_convert_dist_km(d, imp) for d in analysis.distances_km]
        actual = [_convert_speed(s, imp) for s in analysis.actual_speeds]
        predicted = [_convert_speed(s, imp) for s in analysis.predicted_speeds]
        elevations = [_convert_elev(e, imp) for e in analysis.elevations]

        # Speed comparison
        ax1 = self.ax_ride_speed
        ax1.clear()
        _style_ax(ax1, t)
        ax1.plot(distances, actual, color=t["ACCENT"], linewidth=1.2,
                 alpha=0.7, label="Actual")
        ax1.plot(distances, predicted, color=t["ACCENT4"], linewidth=1.2,
                 alpha=0.7, label="Predicted")
        ax1.set_xlabel(f"Distance ({du})", color=t["FG"], fontsize=9)
        ax1.set_ylabel(f"Speed ({su})", color=t["FG"], fontsize=9)
        ax1.set_title("Predicted vs Actual Speed", color=t["FG"],
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"])

        # Elevation profile
        ax2 = self.ax_ride_elev
        ax2.clear()
        _style_ax(ax2, t)
        ax2.fill_between(distances, elevations, alpha=0.3, color=t["ACCENT2"])
        ax2.plot(distances, elevations, color=t["ACCENT2"], linewidth=1.5)
        ax2.set_xlabel(f"Distance ({du})", color=t["FG"], fontsize=9)
        ax2.set_ylabel(f"Elevation ({eu})", color=t["FG"], fontsize=9)
        ax2.set_title("Elevation Profile", color=t["FG"], fontsize=11, fontweight="bold")

        # Error distribution
        ax3 = self.ax_ride_error
        ax3.clear()
        _style_ax(ax3, t)
        errors = [_convert_speed(p - a, imp) for p, a in
                  zip(analysis.predicted_speeds, analysis.actual_speeds)]
        ax3.plot(distances, errors, color=t["ACCENT3"], linewidth=1, alpha=0.7)
        ax3.axhline(0, color=t["FG"], linewidth=0.5, alpha=0.3)
        ax3.fill_between(distances, errors, alpha=0.15, color=t["ACCENT3"])
        ax3.set_xlabel(f"Distance ({du})", color=t["FG"], fontsize=9)
        ax3.set_ylabel(f"Error ({su})", color=t["FG"], fontsize=9)
        ax3.set_title("Prediction Error (Predicted \u2212 Actual)", color=t["FG"],
                      fontsize=11, fontweight="bold")

        self.canvas_ride.draw_idle()

    # ==================================================================
    # TAB 12: What-If Scenarios
    # ==================================================================
    def _build_whatif_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  \U0001f914 What-If  ")

        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left panel — scenario inputs
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        info_card = self._make_card(left, "What-If Analysis")
        ttk.Label(info_card,
                  text="See how changes affect your course time.\n"
                       "Uses the Course Profile segments.",
                  font=("Helvetica", 9), wraplength=280,
                  justify=tk.LEFT).pack(padx=8, pady=4, anchor=tk.W)

        # Weight change
        wt_card = self._make_card(left, "Weight Change (kg)")
        self.var_wi_weight = tk.DoubleVar(value=-2.0)
        self._add_slider(wt_card, "\u0394 Weight", self.var_wi_weight, -10, 10, 0.5, "kg")

        # Power change
        pw_card = self._make_card(left, "Power Change (W)")
        self.var_wi_power = tk.DoubleVar(value=20.0)
        self._add_slider(pw_card, "\u0394 Power", self.var_wi_power, -100, 100, 5, "W")

        # Position change
        pos_card = self._make_card(left, "Riding Position")
        self.var_wi_position = tk.StringVar(value="(no change)")
        positions = ["(no change)"] + [p.value for p in RidingPosition]
        ttk.OptionMenu(pos_card, self.var_wi_position,
                       positions[0], *positions).pack(fill=tk.X, padx=8, pady=4)

        # Tire change
        tire_card = self._make_card(left, "Tire Type")
        self.var_wi_tire = tk.StringVar(value="(no change)")
        tires = ["(no change)"] + [t.value for t in TireType]
        ttk.OptionMenu(tire_card, self.var_wi_tire,
                       tires[0], *tires).pack(fill=tk.X, padx=8, pady=4)

        # Bike weight change
        bw_card = self._make_card(left, "Bike Weight Change (kg)")
        self.var_wi_bike_weight = tk.DoubleVar(value=0.0)
        self._add_slider(bw_card, "\u0394 Bike", self.var_wi_bike_weight, -5, 5, 0.5, "kg")

        ttk.Button(left, text="Run What-If Analysis",
                   command=self._run_whatif).pack(fill=tk.X, padx=8, pady=8)

        # Right panel — results
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        self.lbl_whatif_results = ttk.Label(right, text="Configure changes and click 'Run What-If Analysis'",
                                             font=("Helvetica", 11), wraplength=600,
                                             justify=tk.LEFT)
        self.lbl_whatif_results.pack(padx=12, pady=8, anchor=tk.NW)

        t = self._theme
        self.fig_whatif = Figure(figsize=(8, 5), dpi=100, facecolor=t["BG"])
        self.fig_whatif.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.20)
        self.ax_whatif = self.fig_whatif.add_subplot(111)
        self.canvas_whatif = FigureCanvasTkAgg(self.fig_whatif, master=right)
        self.canvas_whatif.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8)

    def _run_whatif(self) -> None:
        segments = self._get_whatif_segments()
        if not segments:
            messagebox.showwarning("No Course", "Add segments in the Course Profile tab first.")
            return

        rider, bike, _ = self._gather_params()
        changes: list[WhatIfChange] = []

        dw = self.var_wi_weight.get()
        if abs(dw) > 0.1:
            changes.append(WhatIfChange(
                parameter="weight_kg",
                label=f"Weight {dw:+.1f} kg",
                original_value=rider.weight_kg,
                new_value=rider.weight_kg + dw,
            ))

        dp = self.var_wi_power.get()
        if abs(dp) > 0.1:
            changes.append(WhatIfChange(
                parameter="power_watts",
                label=f"Power {dp:+.0f} W",
                original_value=rider.power_watts,
                new_value=rider.power_watts + dp,
            ))

        pos_val = self.var_wi_position.get()
        if pos_val != "(no change)":
            new_pos = next(p for p in RidingPosition if p.value == pos_val)
            changes.append(WhatIfChange(
                parameter="position",
                label=f"Position \u2192 {pos_val}",
                original_value=bike.position,
                new_value=new_pos,
            ))

        tire_val = self.var_wi_tire.get()
        if tire_val != "(no change)":
            new_tire = next(t for t in TireType if t.value == tire_val)
            changes.append(WhatIfChange(
                parameter="tire_type",
                label=f"Tires \u2192 {tire_val}",
                original_value=bike.tire_type,
                new_value=new_tire,
            ))

        dbw = self.var_wi_bike_weight.get()
        if abs(dbw) > 0.1:
            changes.append(WhatIfChange(
                parameter="bike_weight_kg",
                label=f"Bike {dbw:+.1f} kg",
                original_value=bike.weight_kg,
                new_value=bike.weight_kg + dbw,
            ))

        if not changes:
            messagebox.showinfo("No Changes", "Adjust at least one parameter to see its impact.")
            return

        results = what_if_analysis(rider, bike, segments, changes)
        self._last_whatif_results = results
        self._display_whatif_results(results)

    def _get_whatif_segments(self) -> list[CourseSegment]:
        segments: list[CourseSegment] = []
        for sv in self._course_segments:
            segments.append(CourseSegment(
                distance_m=sv["distance"].get(),
                grade_pct=sv["grade"].get(),
                headwind_kmh=sv["wind"].get(),
                wind_direction_deg=sv["wind_dir"].get(),
                elevation_m=sv["elevation"].get(),
                temperature_c=sv["temperature"].get(),
            ))
        return segments

    def _display_whatif_results(self, results: list) -> None:
        imp = self._imperial
        su = _speed_unit(imp)

        text = "What-If Analysis Results:\n\n"
        for r in results:
            sign = "+" if r.time_diff_s > 0 else ""
            speed_sign = "+" if r.speed_diff_kmh > 0 else ""
            spd_diff = _convert_speed(r.speed_diff_kmh, imp)
            text += (
                f"\u2022 {r.change.label}\n"
                f"    Time: {sign}{r.time_diff_s:.1f}s "
                f"({r.pct_improvement:+.2f}%)\n"
                f"    Speed: {speed_sign}{spd_diff:.1f} {su}\n\n"
            )

        base_time = results[0].original_time_s if results else 0
        mins = int(base_time // 60)
        secs = int(base_time % 60)
        text += f"Baseline time: {mins}:{secs:02d}"
        self.lbl_whatif_results.configure(text=text)

        # Chart
        t = self._theme
        ax = self.ax_whatif
        ax.clear()
        _style_ax(ax, t)

        chart_colors = [t[k] for k in CHART_COLOURS_KEYS]
        labels = [r.change.label for r in results]
        times = [r.time_diff_s for r in results]
        colors = [t["SUCCESS"] if td > 0 else t["DANGER"] for td in times]

        bars = ax.barh(labels, times, color=colors, edgecolor=t["BORDER"],
                       alpha=0.85, height=0.5)
        for bar, td in zip(bars, times):
            sign = "+" if td > 0 else ""
            ax.text(bar.get_width() + (0.5 if td >= 0 else -0.5),
                    bar.get_y() + bar.get_height() / 2,
                    f"{sign}{td:.1f}s", ha="left" if td >= 0 else "right",
                    va="center", color=t["FG"], fontsize=9)

        ax.axvline(0, color=t["FG"], linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Time Saved (seconds)", color=t["FG"], fontsize=9)
        ax.set_title("Impact of Each Change", color=t["FG"], fontsize=11, fontweight="bold")
        ax.invert_yaxis()

        self.canvas_whatif.draw_idle()

    # ==================================================================
    # TAB 13: Segment KOM Predictor
    # ==================================================================
    def _build_kom_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  \U0001f3c6 KOM  ")

        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left panel
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        info_card = self._make_card(left, "Segment KOM Predictor")
        ttk.Label(info_card,
                  text="Predict your segment times and compare\n"
                       "against category benchmarks (Cat 5 to Pro).\n"
                       "Uses Course Profile segments.",
                  font=("Helvetica", 9), wraplength=280,
                  justify=tk.LEFT).pack(padx=8, pady=4, anchor=tk.W)

        ttk.Button(info_card, text="Predict KOMs",
                   command=self._run_kom_prediction).pack(fill=tk.X, padx=8, pady=4)

        # Results summary
        summ_card = self._make_card(left, "Overall Standing")
        self.lbl_kom_summary = ttk.Label(summ_card, text="Run prediction to see results",
                                          font=("Helvetica", 10), wraplength=280,
                                          justify=tk.LEFT)
        self.lbl_kom_summary.pack(padx=8, pady=8, anchor=tk.W)

        # Segment details
        detail_card = self._make_card(left, "Segment Details")
        self.lbl_kom_details = ttk.Label(detail_card, text="",
                                          font=("Helvetica", 9), wraplength=280,
                                          justify=tk.LEFT)
        self.lbl_kom_details.pack(padx=8, pady=8, anchor=tk.W)

        # Right panel — charts
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        t = self._theme
        self.fig_kom = Figure(figsize=(8, 8), dpi=100, facecolor=t["BG"])
        self.fig_kom.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.06,
                                     hspace=0.35)
        self.ax_kom_times = self.fig_kom.add_subplot(211)
        self.ax_kom_ranking = self.fig_kom.add_subplot(212)
        self.canvas_kom = FigureCanvasTkAgg(self.fig_kom, master=right)
        self.canvas_kom.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _run_kom_prediction(self) -> None:
        segments = self._get_whatif_segments()
        if not segments:
            messagebox.showwarning("No Course", "Add segments in the Course Profile tab first.")
            return

        rider, bike, _ = self._gather_params()
        prediction = predict_kom(rider, bike, segments)
        self._last_kom_prediction = prediction
        self._display_kom_results(prediction)

    def _display_kom_results(self, prediction: Any) -> None:
        imp = self._imperial
        su = _speed_unit(imp)

        # Overall summary
        rider_total = prediction.total_rider_time_s
        r_mins = int(rider_total // 60)
        r_secs = int(rider_total % 60)

        # Determine overall rank
        overall_rank = "Beginner"
        for cat_name, cat_time in prediction.total_category_times:
            if rider_total <= cat_time:
                overall_rank = cat_name
                break

        summary = f"Your Total Time: {r_mins}:{r_secs:02d}\n"
        summary += f"Overall Ranking: {overall_rank}\n\n"
        summary += "Category Benchmarks:\n"
        for cat_name, cat_time in prediction.total_category_times:
            c_mins = int(cat_time // 60)
            c_secs = int(cat_time % 60)
            diff = rider_total - cat_time
            diff_str = f"+{diff:.0f}s" if diff > 0 else f"{diff:.0f}s"
            marker = " \u25c0" if cat_name == overall_rank else ""
            summary += f"  {cat_name}: {c_mins}:{c_secs:02d} ({diff_str}){marker}\n"

        self.lbl_kom_summary.configure(text=summary)

        # Segment details
        details = ""
        for seg in prediction.segments:
            s_mins = int(seg.rider_time_s // 60)
            s_secs = int(seg.rider_time_s % 60)
            spd = _convert_speed(seg.rider_speed_kmh, imp)
            details += (
                f"Seg {seg.segment_index + 1}: {seg.distance_m:.0f}m "
                f"@ {seg.grade_pct:+.1f}%\n"
                f"  Time: {s_mins}:{s_secs:02d}  "
                f"Speed: {spd:.1f} {su}\n"
                f"  Rank: {seg.rider_rank}\n\n"
            )
        self.lbl_kom_details.configure(text=details.strip())

        self._update_kom_charts(prediction)

    def _update_kom_charts(self, prediction: Any) -> None:
        t = self._theme
        imp = self._imperial

        # Chart 1: Rider time vs category times per segment
        ax1 = self.ax_kom_times
        ax1.clear()
        _style_ax(ax1, t)

        n_segs = len(prediction.segments)
        if n_segs == 0:
            return

        seg_labels = [f"Seg {s.segment_index + 1}\n{s.grade_pct:+.0f}%"
                      for s in prediction.segments]
        x = list(range(n_segs))
        width = 0.12

        # Plot a few key category benchmarks + rider
        categories_to_show = ["World Tour Pro", "Cat 1", "Cat 3", "Cat 5"]
        cat_colors = [t["ACCENT4"], t["ACCENT3"], t["ACCENT2"], t["ACCENT"]]

        for ci, (cat_name, color) in enumerate(zip(categories_to_show, cat_colors)):
            cat_times = []
            for seg in prediction.segments:
                for cn, ct, _ in seg.category_times:
                    if cn == cat_name:
                        cat_times.append(ct)
                        break
            offsets = [xi + (ci - 2) * width for xi in x]
            ax1.bar(offsets, cat_times, width=width, color=color, alpha=0.7,
                    label=cat_name, edgecolor=t["BORDER"], linewidth=0.5)

        # Rider's times
        rider_times = [s.rider_time_s for s in prediction.segments]
        offsets_rider = [xi + 2 * width for xi in x]
        ax1.bar(offsets_rider, rider_times, width=width, color=t["ACCENT4"],
                alpha=1.0, label="You", edgecolor="#ffffff", linewidth=1.5)

        ax1.set_xticks(x)
        ax1.set_xticklabels(seg_labels, fontsize=8)
        ax1.set_ylabel("Time (s)", color=t["FG"], fontsize=9)
        ax1.set_title("Segment Times vs Category Benchmarks", color=t["FG"],
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=7, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"], ncol=5)

        # Chart 2: Overall ranking visualization
        ax2 = self.ax_kom_ranking
        ax2.clear()
        _style_ax(ax2, t)

        cat_names = [c[0] for c in prediction.total_category_times]
        cat_times = [c[1] for c in prediction.total_category_times]

        colors = [t[k] for k in CHART_COLOURS_KEYS]
        bar_colors = [colors[i % len(colors)] for i in range(len(cat_names))]

        bars = ax2.barh(cat_names, cat_times, color=bar_colors, alpha=0.6,
                        edgecolor=t["BORDER"], height=0.6)

        # Add rider's line
        rider_total = prediction.total_rider_time_s
        ax2.axvline(rider_total, color=t["ACCENT4"], linewidth=2.5,
                    linestyle="--", label=f"You: {int(rider_total // 60)}:{int(rider_total % 60):02d}",
                    zorder=10)

        for bar, ct in zip(bars, cat_times):
            m = int(ct // 60)
            s = int(ct % 60)
            ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                     f"{m}:{s:02d}", ha="left", va="center", color=t["FG"], fontsize=8)

        ax2.set_xlabel("Total Time (s)", color=t["FG"], fontsize=9)
        ax2.set_title("Overall Course Ranking", color=t["FG"], fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9, facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                   labelcolor=t["FG"])
        ax2.invert_yaxis()

        self.canvas_kom.draw_idle()

    # ==================================================================
    # TAB 14: Race Planner (Monte Carlo / sweeps / optimization)
    # ==================================================================
    _MC_STD_DEFAULTS = {
        "power_watts": 15.0, "weight_kg": 1.5, "bike_weight_kg": 0.3,
        "headwind_kmh": 8.0, "wind_direction_deg": 20.0,
        "temperature_c": 3.0, "cda": 0.010, "crr": 0.0005,
    }
    _OPT_BOUND_DEFAULTS = {
        "power_watts": (200.0, 350.0), "weight_kg": (60.0, 80.0),
        "bike_weight_kg": (6.0, 9.0), "headwind_kmh": (-10.0, 10.0),
        "wind_direction_deg": (0.0, 360.0), "temperature_c": (5.0, 35.0),
        "cda": (0.20, 0.40), "crr": (0.002, 0.008),
    }

    def _build_planner_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  \U0001f3c1 Race Planner  ")

        self._label_to_key = {
            info["label"]: key for key, info in PLANNER_PARAMS.items()
        }

        sub = ttk.Notebook(tab)
        sub.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._planner_subnb = sub

        self._build_mc_subtab(sub)
        self._build_sweep1d_subtab(sub)
        self._build_sweep2d_subtab(sub)
        self._build_optimize_subtab(sub)

    # ---- helpers ------------------------------------------------------
    def _planner_unit_entry(self, parent: tk.Widget, key: str,
                            init_metric: float, is_delta: bool = False,
                            width: int = 7) -> tuple[tk.DoubleVar, ttk.Entry, ttk.Label]:
        """A numeric entry shown in the active unit; tracked for toggling."""
        disp = _planner_to_display(key, init_metric, self._imperial, is_delta)
        var = tk.DoubleVar(value=round(disp, 5))
        entry = ttk.Entry(parent, textvariable=var, width=width)
        unit_lbl = _label(parent, _planner_unit(key, self._imperial),
                          style="Card.TLabel")
        self._planner_unit_entries.append(
            {"key": key, "var": var, "is_delta": is_delta,
             "unit_label": unit_lbl}
        )
        return var, entry, unit_lbl

    def _read_planner_metric(self, var: tk.DoubleVar, key: str,
                             is_delta: bool = False) -> float:
        return _planner_to_metric(key, var.get(), self._imperial, is_delta)

    def _planner_default_range(self, key: str) -> tuple[float, float]:
        base = None
        try:
            segs = self._get_whatif_segments()
            rider, bike, _ = self._gather_params()
            if segs:
                base = planner_base_value(rider, bike, segs, key)
        except Exception:
            base = None
        if base is not None:
            if key == "power_watts":
                return (max(50.0, base - 80), base + 80)
            if key == "weight_kg":
                return (max(40.0, base - 12), base + 12)
            if key == "bike_weight_kg":
                return (max(4.0, base - 3), base + 3)
            if key == "cda":
                return (max(0.10, base * 0.75), base * 1.25)
        return self._OPT_BOUND_DEFAULTS.get(key, (0.0, 1.0))

    # ---- Monte Carlo --------------------------------------------------
    def _build_mc_subtab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="  Monte Carlo  ")
        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        left = ttk.Frame(paned)
        paned.add(left, weight=1)
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        info = self._make_card(left, "Monte Carlo Finish Time")
        ttk.Label(info,
                  text="Models race-day uncertainty. Each enabled input is\n"
                       "sampled from a Normal(current value, \u00b1\u03c3) per run.\n"
                       "Uses Course Profile segments.",
                  font=("Helvetica", 9), wraplength=300,
                  justify=tk.LEFT).pack(padx=8, pady=4, anchor=tk.W)

        card = self._make_card(left, "Uncertain Inputs (\u00b11\u03c3)")
        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(fill=tk.X, padx=8, pady=4)
        self._mc_enable: dict[str, tk.BooleanVar] = {}
        self._mc_std: dict[str, tk.DoubleVar] = {}
        default_on = {"power_watts", "headwind_kmh"}
        for r, key in enumerate(PLANNER_PARAMS):
            en = tk.BooleanVar(value=key in default_on)
            self._mc_enable[key] = en
            ttk.Checkbutton(grid, text=PLANNER_PARAMS[key]["label"],
                            variable=en, style="TCheckbutton").grid(
                row=r, column=0, sticky=tk.W, pady=1)
            ttk.Label(grid, text="\u00b1", style="Card.TLabel").grid(
                row=r, column=1, padx=(8, 2))
            var, entry, ulbl = self._planner_unit_entry(
                grid, key, self._MC_STD_DEFAULTS[key], is_delta=True)
            self._mc_std[key] = var
            entry.grid(row=r, column=2)
            ulbl.grid(row=r, column=3, sticky=tk.W, padx=(2, 0))

        run_card = self._make_card(left, "Run")
        rrow = ttk.Frame(run_card, style="Card.TFrame")
        rrow.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(rrow, text="Iterations:", style="Card.TLabel").pack(side=tk.LEFT)
        self.var_mc_iters = tk.IntVar(value=1000)
        ttk.Entry(rrow, textvariable=self.var_mc_iters, width=8).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(left, text="Run Monte Carlo",
                   command=self._run_monte_carlo).pack(fill=tk.X, padx=8, pady=8)

        self.lbl_mc_results = ttk.Label(
            right, text="Enable inputs and click 'Run Monte Carlo'.",
            font=("Helvetica", 10), justify=tk.LEFT, wraplength=620)
        self.lbl_mc_results.pack(padx=12, pady=8, anchor=tk.NW)

        t = self._theme
        self.fig_mc = Figure(figsize=(8, 4.6), dpi=100, facecolor=t["BG"])
        self.fig_mc.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.14)
        self.ax_mc = self.fig_mc.add_subplot(111)
        self.canvas_mc = FigureCanvasTkAgg(self.fig_mc, master=right)
        self.canvas_mc.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8)

    def _run_monte_carlo(self) -> None:
        segments = self._get_whatif_segments()
        if not segments:
            messagebox.showwarning("No Course",
                                   "Add segments in the Course Profile tab first.")
            return
        variables: list[MCVariable] = []
        rider, bike, _ = self._gather_params()
        for key in PLANNER_PARAMS:
            if not self._mc_enable[key].get():
                continue
            std = abs(self._read_planner_metric(self._mc_std[key], key, True))
            if std <= 0:
                continue
            mean = planner_base_value(rider, bike, segments, key)
            variables.append(MCVariable(key=key, distribution="normal",
                                        mean=mean, std=std))
        if not variables:
            messagebox.showinfo(
                "No Inputs", "Enable at least one input with a non-zero \u00b1.")
            return
        n = max(10, min(20000, int(self.var_mc_iters.get())))
        try:
            result = monte_carlo_simulation(rider, bike, segments, variables, n=n)
        except Exception as exc:
            messagebox.showerror("Monte Carlo failed", str(exc))
            return
        self._last_mc_result = result
        self._display_mc_results(result)

    def _display_mc_results(self, result: Any) -> None:
        imp = self._imperial
        su = _speed_unit(imp)
        dist_km = result.distance_m / 1000.0
        dist = _convert_dist_km(dist_km, imp)
        p = result.percentiles_s
        mean_spd = _convert_speed(
            result.distance_m / 1000.0 / (result.mean_time_s / 3600.0)
            if result.mean_time_s > 0 else 0.0, imp)
        varied = ", ".join(PLANNER_PARAMS[k]["label"] for k in result.variable_keys)
        text = (
            f"Monte Carlo \u2014 {result.n} runs over {dist:.2f} {_dist_unit(imp)}\n"
            f"Varied: {varied}\n\n"
            f"Mean finish: {_format_time(result.mean_time_s)}  "
            f"(\u00b1{result.std_time_s:.0f}s, avg {mean_spd:.1f} {su})\n"
            f"P5  (best 5%):   {_format_time(p[5])}\n"
            f"P50 (median):    {_format_time(p[50])}\n"
            f"P90:             {_format_time(p[90])}\n"
            f"P95 (worst 5%):  {_format_time(p[95])}\n"
            f"Range: {_format_time(result.min_time_s)} "
            f"\u2192 {_format_time(result.max_time_s)}"
        )
        self.lbl_mc_results.configure(text=text)

        t = self._theme
        ax = self.ax_mc
        ax.clear()
        _style_ax(ax, t)
        times_min = [x / 60.0 for x in result.times_s]
        ax.hist(times_min, bins=40, color=t["ACCENT"], alpha=0.8,
                edgecolor=t["BORDER"])
        for pct, col, lbl in [(50, t["ACCENT2"], "P50"),
                              (90, t["ACCENT3"], "P90"),
                              (95, t["DANGER"], "P95")]:
            ax.axvline(p[pct] / 60.0, color=col, linewidth=1.6,
                       linestyle="--", label=f"{lbl} {_format_time(p[pct])}")
        ax.set_xlabel("Finish time (minutes)", color=t["FG"], fontsize=9)
        ax.set_ylabel("Frequency", color=t["FG"], fontsize=9)
        ax.set_title("Finish-Time Distribution", color=t["FG"],
                     fontsize=11, fontweight="bold")
        ax.legend(facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                  labelcolor=t["FG"], fontsize=8)
        self.canvas_mc.draw_idle()

    # ---- 1D sweep -----------------------------------------------------
    def _build_sweep1d_subtab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="  1D Sweep  ")
        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        left = ttk.Frame(paned)
        paned.add(left, weight=1)
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        info = self._make_card(left, "Parameter Sweep")
        ttk.Label(info,
                  text="Sweep one input across a range and see how your\n"
                       "finish time and average speed respond.",
                  font=("Helvetica", 9), wraplength=300,
                  justify=tk.LEFT).pack(padx=8, pady=4, anchor=tk.W)

        card = self._make_card(left, "Variable")
        self.var_sweep1d_label = tk.StringVar(value=PLANNER_PARAMS["power_watts"]["label"])
        labels = [PLANNER_PARAMS[k]["label"] for k in PLANNER_PARAMS]
        ttk.OptionMenu(card, self.var_sweep1d_label, self.var_sweep1d_label.get(),
                       *labels, command=self._sweep1d_on_var_change).pack(
            fill=tk.X, padx=8, pady=4)

        rng = ttk.Frame(card, style="Card.TFrame")
        rng.pack(fill=tk.X, padx=8, pady=4)
        self._sweep1d_key = "power_watts"
        lo, hi = self._planner_default_range("power_watts")
        ttk.Label(rng, text="Min", style="Card.TLabel").grid(row=0, column=0)
        self.var_sweep1d_min = tk.DoubleVar(
            value=round(_planner_to_display("power_watts", lo, self._imperial), 4))
        ttk.Entry(rng, textvariable=self.var_sweep1d_min, width=8).grid(
            row=0, column=1, padx=4)
        ttk.Label(rng, text="Max", style="Card.TLabel").grid(row=0, column=2)
        self.var_sweep1d_max = tk.DoubleVar(
            value=round(_planner_to_display("power_watts", hi, self._imperial), 4))
        ttk.Entry(rng, textvariable=self.var_sweep1d_max, width=8).grid(
            row=0, column=3, padx=4)
        self.lbl_sweep1d_unit = _label(rng, _planner_unit("power_watts", self._imperial),
                                       style="Card.TLabel")
        self.lbl_sweep1d_unit.grid(row=0, column=4, padx=(4, 0))

        prow = ttk.Frame(card, style="Card.TFrame")
        prow.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(prow, text="Points:", style="Card.TLabel").pack(side=tk.LEFT)
        self.var_sweep1d_points = tk.IntVar(value=25)
        ttk.Entry(prow, textvariable=self.var_sweep1d_points, width=6).pack(
            side=tk.LEFT, padx=6)

        ttk.Button(left, text="Run Sweep",
                   command=self._run_sweep1d).pack(fill=tk.X, padx=8, pady=8)

        self.lbl_sweep1d_results = ttk.Label(
            right, text="Pick a variable and click 'Run Sweep'.",
            font=("Helvetica", 10), justify=tk.LEFT, wraplength=620)
        self.lbl_sweep1d_results.pack(padx=12, pady=8, anchor=tk.NW)

        t = self._theme
        self.fig_sweep1d = Figure(figsize=(8, 4.6), dpi=100, facecolor=t["BG"])
        self.fig_sweep1d.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.14)
        self.ax_sweep1d = self.fig_sweep1d.add_subplot(111)
        self.canvas_sweep1d = FigureCanvasTkAgg(self.fig_sweep1d, master=right)
        self.canvas_sweep1d.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8)

    def _sweep1d_on_var_change(self, _label: str | None = None) -> None:
        key = self._label_to_key[self.var_sweep1d_label.get()]
        self._sweep1d_key = key
        lo, hi = self._planner_default_range(key)
        self.var_sweep1d_min.set(round(_planner_to_display(key, lo, self._imperial), 4))
        self.var_sweep1d_max.set(round(_planner_to_display(key, hi, self._imperial), 4))
        self.lbl_sweep1d_unit.configure(text=_planner_unit(key, self._imperial))

    def _run_sweep1d(self) -> None:
        segments = self._get_whatif_segments()
        if not segments:
            messagebox.showwarning("No Course",
                                   "Add segments in the Course Profile tab first.")
            return
        key = self._sweep1d_key
        lo = _planner_to_metric(key, self.var_sweep1d_min.get(), self._imperial)
        hi = _planner_to_metric(key, self.var_sweep1d_max.get(), self._imperial)
        npts = max(2, min(200, int(self.var_sweep1d_points.get())))
        rider, bike, _ = self._gather_params()
        try:
            result = sweep_1d(rider, bike, segments, key, lo, hi, npts)
        except Exception as exc:
            messagebox.showerror("Sweep failed", str(exc))
            return
        self._last_sweep1d_result = result
        self._display_sweep1d_results(result)

    def _display_sweep1d_results(self, result: Any) -> None:
        imp = self._imperial
        key = result.key
        unit = _planner_unit(key, imp)
        x = [_planner_to_display(key, v, imp) for v in result.values]
        times_min = [tt / 60.0 for tt in result.times_s]
        speeds = [_convert_speed(s, imp) for s in result.avg_speeds_kmh]
        su = _speed_unit(imp)

        best_i = int(min(range(len(result.times_s)),
                         key=lambda i: result.times_s[i]))
        worst_i = int(max(range(len(result.times_s)),
                          key=lambda i: result.times_s[i]))
        spread = result.times_s[worst_i] - result.times_s[best_i]
        text = (
            f"Sweep: {PLANNER_PARAMS[key]['label']}\n\n"
            f"Range: {x[0]:.3g} \u2192 {x[-1]:.3g} {unit}\n"
            f"Fastest: {_format_time(result.times_s[best_i])} "
            f"at {x[best_i]:.3g} {unit}\n"
            f"Slowest: {_format_time(result.times_s[worst_i])} "
            f"at {x[worst_i]:.3g} {unit}\n"
            f"Total spread: {spread:.0f}s across the range"
        )
        self.lbl_sweep1d_results.configure(text=text)

        t = self._theme
        ax = self.ax_sweep1d
        ax.clear()
        _style_ax(ax, t)
        ax.plot(x, times_min, color=t["ACCENT"], linewidth=2, marker="o",
                markersize=3, label="Finish time")
        ax.set_xlabel(f"{PLANNER_PARAMS[key]['label']} ({unit})",
                      color=t["FG"], fontsize=9)
        ax.set_ylabel("Finish time (min)", color=t["ACCENT"], fontsize=9)
        ax.set_title(f"Finish Time vs {PLANNER_PARAMS[key]['label']}",
                     color=t["FG"], fontsize=11, fontweight="bold")

        # current value marker
        base_disp = _planner_to_display(key, result.base_value, imp)
        if x[0] <= base_disp <= x[-1] or x[-1] <= base_disp <= x[0]:
            ax.axvline(base_disp, color=t["FG_DIM"], linestyle=":",
                       linewidth=1.2, label="Current")

        ax2 = ax.twinx()
        ax2.plot(x, speeds, color=t["ACCENT2"], linewidth=1.6,
                 linestyle="--", label=f"Avg speed ({su})")
        ax2.set_ylabel(f"Avg speed ({su})", color=t["ACCENT2"], fontsize=9)
        ax2.tick_params(colors=t["FG"], labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color(t["BORDER"])

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                  labelcolor=t["FG"], fontsize=8, loc="best")
        self.fig_sweep1d.tight_layout()
        self.canvas_sweep1d.draw_idle()

    # ---- 2D contour ---------------------------------------------------
    def _build_sweep2d_subtab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="  2D Contour  ")
        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        left = ttk.Frame(paned)
        paned.add(left, weight=1)
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        info = self._make_card(left, "2D Sweep / Contour")
        ttk.Label(info,
                  text="Sweep two inputs together to map finish time as a\n"
                       "contour. The optimum cell and your current setup\n"
                       "are marked.",
                  font=("Helvetica", 9), wraplength=300,
                  justify=tk.LEFT).pack(padx=8, pady=4, anchor=tk.W)

        labels = [PLANNER_PARAMS[k]["label"] for k in PLANNER_PARAMS]

        # X axis
        xcard = self._make_card(left, "X Axis")
        self.var_sweep2d_xlabel = tk.StringVar(value=PLANNER_PARAMS["power_watts"]["label"])
        self._sweep2d_xkey = "power_watts"
        ttk.OptionMenu(xcard, self.var_sweep2d_xlabel, self.var_sweep2d_xlabel.get(),
                       *labels, command=lambda v: self._sweep2d_on_var_change("x")).pack(
            fill=tk.X, padx=8, pady=2)
        xr = ttk.Frame(xcard, style="Card.TFrame")
        xr.pack(fill=tk.X, padx=8, pady=2)
        lo, hi = self._planner_default_range("power_watts")
        ttk.Label(xr, text="Min", style="Card.TLabel").grid(row=0, column=0)
        self.var_sweep2d_xmin = tk.DoubleVar(value=round(_planner_to_display("power_watts", lo, self._imperial), 4))
        ttk.Entry(xr, textvariable=self.var_sweep2d_xmin, width=7).grid(row=0, column=1, padx=3)
        ttk.Label(xr, text="Max", style="Card.TLabel").grid(row=0, column=2)
        self.var_sweep2d_xmax = tk.DoubleVar(value=round(_planner_to_display("power_watts", hi, self._imperial), 4))
        ttk.Entry(xr, textvariable=self.var_sweep2d_xmax, width=7).grid(row=0, column=3, padx=3)
        self.lbl_sweep2d_xunit = _label(xr, _planner_unit("power_watts", self._imperial), style="Card.TLabel")
        self.lbl_sweep2d_xunit.grid(row=0, column=4, padx=(3, 0))

        # Y axis
        ycard = self._make_card(left, "Y Axis")
        self.var_sweep2d_ylabel = tk.StringVar(value=PLANNER_PARAMS["weight_kg"]["label"])
        self._sweep2d_ykey = "weight_kg"
        ttk.OptionMenu(ycard, self.var_sweep2d_ylabel, self.var_sweep2d_ylabel.get(),
                       *labels, command=lambda v: self._sweep2d_on_var_change("y")).pack(
            fill=tk.X, padx=8, pady=2)
        yr = ttk.Frame(ycard, style="Card.TFrame")
        yr.pack(fill=tk.X, padx=8, pady=2)
        lo, hi = self._planner_default_range("weight_kg")
        ttk.Label(yr, text="Min", style="Card.TLabel").grid(row=0, column=0)
        self.var_sweep2d_ymin = tk.DoubleVar(value=round(_planner_to_display("weight_kg", lo, self._imperial), 4))
        ttk.Entry(yr, textvariable=self.var_sweep2d_ymin, width=7).grid(row=0, column=1, padx=3)
        ttk.Label(yr, text="Max", style="Card.TLabel").grid(row=0, column=2)
        self.var_sweep2d_ymax = tk.DoubleVar(value=round(_planner_to_display("weight_kg", hi, self._imperial), 4))
        ttk.Entry(yr, textvariable=self.var_sweep2d_ymax, width=7).grid(row=0, column=3, padx=3)
        self.lbl_sweep2d_yunit = _label(yr, _planner_unit("weight_kg", self._imperial), style="Card.TLabel")
        self.lbl_sweep2d_yunit.grid(row=0, column=4, padx=(3, 0))

        gcard = self._make_card(left, "Grid")
        grow = ttk.Frame(gcard, style="Card.TFrame")
        grow.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(grow, text="Resolution:", style="Card.TLabel").pack(side=tk.LEFT)
        self.var_sweep2d_res = tk.IntVar(value=22)
        ttk.Entry(grow, textvariable=self.var_sweep2d_res, width=6).pack(side=tk.LEFT, padx=6)
        ttk.Label(grow, text="(per axis)", style="Small.TLabel").pack(side=tk.LEFT)

        ttk.Button(left, text="Run Contour",
                   command=self._run_sweep2d).pack(fill=tk.X, padx=8, pady=8)

        self.lbl_sweep2d_results = ttk.Label(
            right, text="Choose X and Y inputs and click 'Run Contour'.",
            font=("Helvetica", 10), justify=tk.LEFT, wraplength=620)
        self.lbl_sweep2d_results.pack(padx=12, pady=8, anchor=tk.NW)

        t = self._theme
        self.fig_sweep2d = Figure(figsize=(8, 4.8), dpi=100, facecolor=t["BG"])
        self.fig_sweep2d.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.14)
        self.ax_sweep2d = self.fig_sweep2d.add_subplot(111)
        self.canvas_sweep2d = FigureCanvasTkAgg(self.fig_sweep2d, master=right)
        self.canvas_sweep2d.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8)
        self._sweep2d_colorbar = None

    def _sweep2d_on_var_change(self, axis: str) -> None:
        if axis == "x":
            key = self._label_to_key[self.var_sweep2d_xlabel.get()]
            self._sweep2d_xkey = key
            lo, hi = self._planner_default_range(key)
            self.var_sweep2d_xmin.set(round(_planner_to_display(key, lo, self._imperial), 4))
            self.var_sweep2d_xmax.set(round(_planner_to_display(key, hi, self._imperial), 4))
            self.lbl_sweep2d_xunit.configure(text=_planner_unit(key, self._imperial))
        else:
            key = self._label_to_key[self.var_sweep2d_ylabel.get()]
            self._sweep2d_ykey = key
            lo, hi = self._planner_default_range(key)
            self.var_sweep2d_ymin.set(round(_planner_to_display(key, lo, self._imperial), 4))
            self.var_sweep2d_ymax.set(round(_planner_to_display(key, hi, self._imperial), 4))
            self.lbl_sweep2d_yunit.configure(text=_planner_unit(key, self._imperial))

    def _run_sweep2d(self) -> None:
        segments = self._get_whatif_segments()
        if not segments:
            messagebox.showwarning("No Course",
                                   "Add segments in the Course Profile tab first.")
            return
        kx, ky = self._sweep2d_xkey, self._sweep2d_ykey
        if kx == ky:
            messagebox.showinfo("Pick two", "Choose two different variables.")
            return
        xr = (_planner_to_metric(kx, self.var_sweep2d_xmin.get(), self._imperial),
              _planner_to_metric(kx, self.var_sweep2d_xmax.get(), self._imperial))
        yr = (_planner_to_metric(ky, self.var_sweep2d_ymin.get(), self._imperial),
              _planner_to_metric(ky, self.var_sweep2d_ymax.get(), self._imperial))
        res = max(5, min(60, int(self.var_sweep2d_res.get())))
        rider, bike, _ = self._gather_params()
        try:
            result = sweep_2d(rider, bike, segments, kx, ky, xr, yr, res, res)
        except Exception as exc:
            messagebox.showerror("Contour failed", str(exc))
            return
        self._last_sweep2d_result = result
        self._display_sweep2d_results(result)

    def _display_sweep2d_results(self, result: Any) -> None:
        imp = self._imperial
        kx, ky = result.key_x, result.key_y
        ux, uy = _planner_unit(kx, imp), _planner_unit(ky, imp)
        x = [_planner_to_display(kx, v, imp) for v in result.x_values]
        y = [_planner_to_display(ky, v, imp) for v in result.y_values]
        z_min = [[tt / 60.0 for tt in row] for row in result.times_s]
        best_x = _planner_to_display(kx, result.best_x, imp)
        best_y = _planner_to_display(ky, result.best_y, imp)
        base_x = _planner_to_display(kx, result.base_x, imp)
        base_y = _planner_to_display(ky, result.base_y, imp)

        text = (
            f"X: {PLANNER_PARAMS[kx]['label']}   Y: {PLANNER_PARAMS[ky]['label']}\n\n"
            f"Best finish: {_format_time(result.best_time_s)}\n"
            f"  at {PLANNER_PARAMS[kx]['label']} = {best_x:.3g} {ux}\n"
            f"  and {PLANNER_PARAMS[ky]['label']} = {best_y:.3g} {uy}"
        )
        self.lbl_sweep2d_results.configure(text=text)

        import numpy as _np
        t = self._theme
        if self._sweep2d_colorbar is not None:
            try:
                self._sweep2d_colorbar.remove()
            except Exception:
                pass
            self._sweep2d_colorbar = None
        self.fig_sweep2d.clf()
        ax = self.fig_sweep2d.add_subplot(111)
        self.ax_sweep2d = ax
        _style_ax(ax, t)
        X, Y = _np.meshgrid(_np.array(x), _np.array(y))
        Z = _np.array(z_min)
        cf = ax.contourf(X, Y, Z, levels=18, cmap="viridis")
        ax.contour(X, Y, Z, levels=10, colors=t["BG"], linewidths=0.4, alpha=0.4)
        cb = self.fig_sweep2d.colorbar(cf, ax=ax)
        cb.set_label("Finish time (min)", color=t["FG"], fontsize=9)
        cb.ax.tick_params(colors=t["FG"], labelsize=8)
        self._sweep2d_colorbar = cb

        ax.scatter([best_x], [best_y], marker="*", s=240, color=t["ACCENT2"],
                   edgecolor="black", zorder=5, label="Optimum")
        if (min(x) <= base_x <= max(x)) and (min(y) <= base_y <= max(y)):
            ax.scatter([base_x], [base_y], marker="o", s=70, color=t["DANGER"],
                       edgecolor="white", zorder=5, label="Current")
        ax.set_xlabel(f"{PLANNER_PARAMS[kx]['label']} ({ux})",
                      color=t["FG"], fontsize=9)
        ax.set_ylabel(f"{PLANNER_PARAMS[ky]['label']} ({uy})",
                      color=t["FG"], fontsize=9)
        ax.set_title("Finish-Time Contour", color=t["FG"],
                     fontsize=11, fontweight="bold")
        ax.legend(facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                  labelcolor=t["FG"], fontsize=8, loc="best")
        self.canvas_sweep2d.draw_idle()

    # ---- Optimize -----------------------------------------------------
    def _build_optimize_subtab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="  Optimize  ")
        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        left = ttk.Frame(paned)
        paned.add(left, weight=1)
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        info = self._make_card(left, "Optimize Setup")
        ttk.Label(info,
                  text="Finds the fastest setup within your bounds, then\n"
                       "recommends a pacing plan for the result.",
                  font=("Helvetica", 9), wraplength=300,
                  justify=tk.LEFT).pack(padx=8, pady=4, anchor=tk.W)

        card = self._make_card(left, "Variables & Bounds")
        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(grid, text="", style="Card.TLabel").grid(row=0, column=0)
        ttk.Label(grid, text="Min", style="Card.TLabel").grid(row=0, column=2)
        ttk.Label(grid, text="Max", style="Card.TLabel").grid(row=0, column=3)
        self._opt_enable: dict[str, tk.BooleanVar] = {}
        self._opt_low: dict[str, tk.DoubleVar] = {}
        self._opt_high: dict[str, tk.DoubleVar] = {}
        default_on = {"power_watts", "weight_kg", "cda"}
        for r, key in enumerate(PLANNER_PARAMS, start=1):
            en = tk.BooleanVar(value=key in default_on)
            self._opt_enable[key] = en
            ttk.Checkbutton(grid, text=PLANNER_PARAMS[key]["label"],
                            variable=en, style="TCheckbutton").grid(
                row=r, column=0, sticky=tk.W, pady=1)
            lo, hi = self._OPT_BOUND_DEFAULTS[key]
            lvar, lentry, _lu = self._planner_unit_entry(grid, key, lo)
            hvar, hentry, hu = self._planner_unit_entry(grid, key, hi)
            self._opt_low[key] = lvar
            self._opt_high[key] = hvar
            lentry.grid(row=r, column=2, padx=2)
            hentry.grid(row=r, column=3, padx=2)
            hu.grid(row=r, column=4, sticky=tk.W, padx=(2, 0))

        ttk.Button(left, text="Optimize",
                   command=self._run_optimize).pack(fill=tk.X, padx=8, pady=8)

        self.lbl_optimize_results = ttk.Label(
            right, text="Select variables, set bounds, and click 'Optimize'.",
            font=("Helvetica", 10), justify=tk.LEFT, wraplength=640)
        self.lbl_optimize_results.pack(padx=12, pady=8, anchor=tk.NW)

        t = self._theme
        self.fig_optimize = Figure(figsize=(8, 4.0), dpi=100, facecolor=t["BG"])
        self.fig_optimize.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.18)
        self.ax_optimize = self.fig_optimize.add_subplot(111)
        self.canvas_optimize = FigureCanvasTkAgg(self.fig_optimize, master=right)
        self.canvas_optimize.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8)

    def _run_optimize(self) -> None:
        segments = self._get_whatif_segments()
        if not segments:
            messagebox.showwarning("No Course",
                                   "Add segments in the Course Profile tab first.")
            return
        variables: list[OptimizeVariable] = []
        for key in PLANNER_PARAMS:
            if not self._opt_enable[key].get():
                continue
            lo = _planner_to_metric(key, self._opt_low[key].get(), self._imperial)
            hi = _planner_to_metric(key, self._opt_high[key].get(), self._imperial)
            variables.append(OptimizeVariable(key=key, low=lo, high=hi))
        if not variables:
            messagebox.showinfo("No Variables", "Enable at least one variable.")
            return
        rider, bike, _ = self._gather_params()
        try:
            result = optimize_setup(rider, bike, segments, variables,
                                    ftp=self.var_ftp.get(), include_pacing=True)
        except Exception as exc:
            messagebox.showerror("Optimize failed", str(exc))
            return
        self._last_optimize_result = result
        self._display_optimize_results(result)

    def _display_optimize_results(self, result: Any) -> None:
        imp = self._imperial
        lines = [
            f"Baseline finish:  {_format_time(result.baseline_time_s)}",
            f"Optimized finish: {_format_time(result.optimized_time_s)}",
            f"Time saved:       {result.time_saved_s:.0f}s "
            f"({result.time_saved_s / 60.0:.1f} min)",
            "",
            "Recommended setup:",
        ]
        for key in result.variable_keys:
            unit = _planner_unit(key, imp)
            base = _planner_to_display(key, result.baseline_values[key], imp)
            opt = _planner_to_display(key, result.optimized_values[key], imp)
            lines.append(
                f"  \u2022 {PLANNER_PARAMS[key]['label']}: "
                f"{base:.3g} \u2192 {opt:.3g} {unit}")
        if result.pacing is not None:
            pac = result.pacing
            lines.append("")
            lines.append(
                f"Pacing: avg {pac.avg_power:.0f} W, "
                f"{_format_time(pac.total_time_s)} "
                f"(saves {pac.time_saved_s:.0f}s vs even pace)")
        self.lbl_optimize_results.configure(text="\n".join(lines))

        t = self._theme
        ax = self.ax_optimize
        ax.clear()
        _style_ax(ax, t)
        if result.pacing is not None and result.pacing.segments:
            segs = result.pacing.segments
            idx = list(range(1, len(segs) + 1))
            powers = [s.optimal_power for s in segs]
            colors = [t["DANGER"] if s.grade_pct > 2 else
                      (t["ACCENT3"] if s.grade_pct >= -2 else t["ACCENT2"])
                      for s in segs]
            ax.bar(idx, powers, color=colors, edgecolor=t["BORDER"], alpha=0.9)
            ax.axhline(result.pacing.avg_power, color=t["FG_DIM"],
                       linestyle="--", linewidth=1, label="Avg power")
            ax.set_xlabel("Segment", color=t["FG"], fontsize=9)
            ax.set_ylabel("Recommended power (W)", color=t["FG"], fontsize=9)
            ax.set_title("Recommended Pacing (red=climb, green=descent)",
                         color=t["FG"], fontsize=11, fontweight="bold")
            ax.legend(facecolor=t["BG_LIGHT"], edgecolor=t["BORDER"],
                      labelcolor=t["FG"], fontsize=8)
        else:
            ax.bar(["Baseline", "Optimized"],
                   [result.baseline_time_s / 60.0,
                    result.optimized_time_s / 60.0],
                   color=[t["DANGER"], t["ACCENT2"]], edgecolor=t["BORDER"])
            ax.set_ylabel("Finish time (min)", color=t["FG"], fontsize=9)
            ax.set_title("Baseline vs Optimized", color=t["FG"],
                         fontsize=11, fontweight="bold")
        self.canvas_optimize.draw_idle()

    def _reunit_planner(self, old_imp: bool, new_imp: bool) -> None:
        """Convert all tracked planner entries between unit systems."""
        for rec in self._planner_unit_entries:
            metric = _planner_to_metric(rec["key"], rec["var"].get(),
                                        old_imp, rec["is_delta"])
            rec["var"].set(round(_planner_to_display(
                rec["key"], metric, new_imp, rec["is_delta"]), 5))
            rec["unit_label"].configure(
                text=_planner_unit(rec["key"], new_imp))
        # Sweep 1D min/max + unit label
        for key, vmin, vmax, ulbl in [
            (self._sweep1d_key, self.var_sweep1d_min, self.var_sweep1d_max,
             self.lbl_sweep1d_unit),
            (self._sweep2d_xkey, self.var_sweep2d_xmin, self.var_sweep2d_xmax,
             self.lbl_sweep2d_xunit),
            (self._sweep2d_ykey, self.var_sweep2d_ymin, self.var_sweep2d_ymax,
             self.lbl_sweep2d_yunit),
        ]:
            for v in (vmin, vmax):
                m = _planner_to_metric(key, v.get(), old_imp)
                v.set(round(_planner_to_display(key, m, new_imp), 4))
            ulbl.configure(text=_planner_unit(key, new_imp))
        # Re-render cached charts
        if self._last_mc_result is not None:
            self._display_mc_results(self._last_mc_result)
        if self._last_sweep1d_result is not None:
            self._display_sweep1d_results(self._last_sweep1d_result)
        if self._last_sweep2d_result is not None:
            self._display_sweep2d_results(self._last_sweep2d_result)
        if self._last_optimize_result is not None:
            self._display_optimize_results(self._last_optimize_result)

    # ==================================================================
    # TAB 15: Export
    # ==================================================================
    def _build_export_tab(self) -> None:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  \U0001f4be Export  ")

        info = ttk.Frame(tab)
        info.pack(fill=tk.X, padx=20, pady=20)

        _label(info, "Export current simulation or course profile results.",
               font=("Helvetica", 12)).pack(anchor=tk.W, pady=4)

        btn_frame = ttk.Frame(info)
        btn_frame.pack(anchor=tk.W, pady=8)

        buttons = [
            ("Export Current Result to CSV", self._export_result_csv),
            ("Export Course Profile to CSV", self._export_course_csv),
            ("Generate PDF Report", self._export_pdf),
        ]
        for text, cmd in buttons:
            ttk.Button(btn_frame, text=text, style="Secondary.TButton",
                       command=cmd).pack(anchor=tk.W, pady=3)

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

    def _export_pdf(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not path:
            return

        t = self._theme
        imp = self._imperial
        su = _speed_unit(imp)
        du = _dist_unit(imp)
        eu = _elev_unit(imp)
        rider, bike, course = self._gather_params()
        result = solve_speed(rider, bike, course)

        with PdfPages(path) as pdf:
            # Page 1: Main simulation
            fig1 = Figure(figsize=(11, 8.5), dpi=150, facecolor="white")
            fig1.suptitle("Bike Power Simulator Report", fontsize=16, fontweight="bold")

            ax_info = fig1.add_subplot(311)
            ax_info.axis("off")
            result_spd = _convert_speed(result.speed_kmh, imp)
            info_text = (
                f"Power: {rider.power_watts:.0f} W  |  "
                f"Weight: {rider.weight_kg:.1f} kg  |  "
                f"Height: {rider.height_cm:.0f} cm\n"
                f"Bike: {bike.weight_kg:.1f} kg  |  "
                f"Tire: {bike.tire_type.value}  |  "
                f"Position: {bike.position.value}\n"
                f"Grade: {course.grade_pct:.1f}%  |  "
                f"Wind: {course.headwind_kmh:.0f} km/h\n\n"
                f"RESULT: {result_spd:.1f} {su}\n"
                f"Aero: {result.power_aero:.1f} W  |  "
                f"Rolling: {result.power_rolling:.1f} W  |  "
                f"Gravity: {result.power_gravity:.1f} W"
            )
            ax_info.text(0.05, 0.5, info_text, transform=ax_info.transAxes,
                        fontsize=10, verticalalignment="center", family="monospace")

            ax_curve = fig1.add_subplot(312)
            powers, speeds_raw = speed_vs_power_curve(rider, bike, course)
            speeds_disp = [_convert_speed(s, imp) for s in speeds_raw]
            ax_curve.plot(powers, speeds_disp, color="steelblue", linewidth=2)
            ax_curve.axvline(rider.power_watts, color="red", linestyle="--", linewidth=1)
            ax_curve.scatter([rider.power_watts], [result_spd], color="green", s=60, zorder=5)
            ax_curve.set_xlabel("Power (W)")
            ax_curve.set_ylabel(f"Speed ({su})")
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

                dk = [_convert_dist_km(d / 1000, imp) for d in cr.distances_cumulative]
                cr_speeds = [_convert_speed(s, imp) for s in cr.speeds]
                cr_avg = _convert_speed(cr.avg_speed_kmh, imp)
                ax_s = fig2.add_subplot(211)
                mid_d = [(dk[i] + dk[i + 1]) / 2 for i in range(len(cr_speeds))]
                ax_s.bar(mid_d, cr_speeds, width=[dk[i + 1] - dk[i] for i in range(len(cr_speeds))],
                         color="steelblue", alpha=0.8)
                ax_s.axhline(cr_avg, color="red", linestyle="--")
                ax_s.set_xlabel(f"Distance ({du})")
                ax_s.set_ylabel(f"Speed ({su})")
                ax_s.set_title("Speed by Segment")
                ax_s.grid(True, alpha=0.3)

                cr_elevations = [_convert_elev(e, imp) for e in cr.elevations]
                ax_e = fig2.add_subplot(212)
                ax_e.fill_between(dk, cr_elevations, alpha=0.3, color="green")
                ax_e.plot(dk, cr_elevations, color="green", linewidth=2)
                ax_e.set_xlabel(f"Distance ({du})")
                ax_e.set_ylabel(f"Elevation ({eu})")
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
        unit: str = "",
    ) -> dict[str, Any]:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill=tk.X, padx=10, pady=2)

        lbl_text = f"{label}:"
        lbl_widget = _label(row, lbl_text, style="Card.TLabel")
        lbl_widget.pack(side=tk.LEFT)

        # Value + unit on the right
        val_frame = ttk.Frame(row, style="Card.TFrame")
        val_frame.pack(side=tk.RIGHT)
        val_label = _label(val_frame, f"{var.get():.1f}", style="Card.TLabel")
        val_label.pack(side=tk.LEFT, padx=(4, 0))
        if unit:
            unit_lbl = _label(val_frame, f" {unit}", style="Card.TLabel")
            unit_lbl.pack(side=tk.LEFT)

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
        old_imperial = self._imperial
        self._imperial = not self._imperial
        if self._imperial:
            self._unit_btn.configure(text="\u21c4 Metric")
            self.lbl_speed_unit1.configure(text="mph")
            self.lbl_speed_unit2.configure(text="km/h")
        else:
            self._unit_btn.configure(text="\u21c4 Imperial")
            self.lbl_speed_unit1.configure(text="km/h")
            self.lbl_speed_unit2.configure(text="mph")
        self._run_simulation()
        # Re-render other tabs that have cached data
        if self._last_course_result:
            self._run_course_profile()
        if self._ride_analysis is not None:
            self._run_ride_analysis()
        if hasattr(self, '_last_whatif_results') and self._last_whatif_results:
            self._display_whatif_results(self._last_whatif_results)
        if hasattr(self, '_last_kom_prediction') and self._last_kom_prediction:
            self._display_kom_results(self._last_kom_prediction)
        # Race Planner: convert entries + re-render cached charts
        self._reunit_planner(old_imperial, self._imperial)

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
        self._last_result = result

        if self._imperial:
            self.lbl_speed_primary.configure(text=f"{result.speed_mph:.1f}")
            self.lbl_speed_secondary.configure(text=f"{result.speed_kmh:.1f}")
        else:
            self.lbl_speed_primary.configure(text=f"{result.speed_kmh:.1f}")
            self.lbl_speed_secondary.configure(text=f"{result.speed_mph:.1f}")

        # Calorie estimate for 1 hour at current power
        cal_1hr = estimate_calories(rider.power_watts, 3600)
        self._metric_cal["value"].configure(text=f"{cal_1hr:.0f}")

        # Power zone
        ftp = self.var_ftp.get()
        zone_name, zone_color = zone_for_power(rider.power_watts, ftp)
        self._metric_zone["value"].configure(text=zone_name)

        # W/kg on main tab
        w_kg = power_to_weight_ratio(ftp, rider.weight_kg)
        cat = classify_rider(w_kg)
        self._metric_wkg["value"].configure(text=f"{w_kg:.2f}")
        self._metric_wkg["label"].configure(text=f"W/KG ({cat})")

        details = (
            f"Air density: {result.air_density:.3f} kg/m\u00b3  \u2502  "
            f"CdA: {result.cda:.4f} m\u00b2  \u2502  Crr: {result.crr:.4f}\n"
            f"Aero: {result.power_aero:.1f} W  \u2502  "
            f"Rolling: {result.power_rolling:.1f} W  \u2502  "
            f"Gravity: {result.power_gravity:.1f} W  \u2502  "
            f"Drivetrain loss: {result.power_drivetrain_loss:.1f} W"
        )
        self.lbl_details.configure(text=details)
        self._update_charts(rider, bike, course, result)
        self._update_status_bar(result, rider)

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
