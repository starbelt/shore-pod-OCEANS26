"""
Shore Pod Energy Harvesting — Operating Strategy Simulation

Produces four paper figures (all saved to plots/):
  1. buffer_sizing.png      — minimum buffer capacity sizing procedure
  2. heatmap.png            — uptime vs. hibernation threshold sweep (Figure 9)
  3. stacked_strategies.png — per-strategy buffer energy traces (Figure 8)
  4. uptime_percent.png     — uptime % of maximum by strategy & environment (Figure 10)

Usage:
    python threshold_study.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, replace
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (
    STRATEGIES, GRID_CASES,
    seattle_lat, seattle_long,
    shore_pod_realistic_SA_m2, sb_sca_efficiency,
)
from irradiance import get_irradiance_data
from buffer_sizing import compute_required_buffer_max_accumulated_floor0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SiteInputs:
    latitude: float
    longitude: float
    start_date: str   # "YYYY-MM-DD"
    end_date: str     # "YYYY-MM-DD"
    location: str = "site"
    dt_hours: float = 1.0


@dataclass
class HardwareInputs:
    active_area_m2: float
    efficiency: float
    p_active_W: float
    p_hibernate_W: float
    buffer_capacity_J: float
    buffer_init_J: float = 2000.0


# Hardware voltage-protection thresholds — enforced in all simulations
VOLTAGE_LOW_J  = 1000.0   # below this the device enters hard sleep (UVLO)
VOLTAGE_HIGH_J = 7000.0   # above this the device may wake from hard sleep


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------

def _save_or_show_figure(fig, show_plot=True, save_dir=None, filename=None, dpi=300):
    if save_dir and filename:
        os.makedirs(save_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(filename))
        if not safe.lower().endswith(".png"):
            safe += ".png"
        path = os.path.join(save_dir, safe)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Solar power
# ---------------------------------------------------------------------------

def compute_hourly_solar_power_W(site: SiteInputs, hw: HardwareInputs):
    """Fetch hourly irradiance from Open-Meteo and convert to solar electrical power (W)."""
    df, _, dni_hourly, _ = get_irradiance_data(
        site.latitude, site.longitude, site.start_date, site.end_date, site.location
    )
    solar_W = np.asarray(dni_hourly, dtype=float) * hw.active_area_m2 * hw.efficiency
    time = df["time"].to_numpy() if "time" in df.columns else df.index.to_numpy()
    return time, solar_W, dni_hourly


# ---------------------------------------------------------------------------
# Simulation engines
# ---------------------------------------------------------------------------

def simulate_hibernation_strategy(
    solar_power_W,
    hw: HardwareInputs,
    enter_threshold_J: float,
    exit_threshold_J: float,
    initial_state: str = "active",
    dt_hours: float = 1.0,
):
    """
    Schmitt-trigger hibernation strategy.

    Device enters sleep when buffer ≤ enter_threshold_J and wakes when
    buffer ≥ exit_threshold_J.

    Returns:
        uptime_hours (float), duty (0/1 array), buffer_J (energy trace)
    """
    solar_power_W = np.asarray(solar_power_W, dtype=float)
    n    = len(solar_power_W)
    dt_s = float(dt_hours) * 3600.0
    cap  = float(hw.buffer_capacity_J)
    enter = float(max(0.0, enter_threshold_J))
    exit_ = float(max(0.0, exit_threshold_J))
    if exit_ < enter:
        enter, exit_ = exit_, enter

    buffer_J  = np.zeros(n, dtype=float)
    duty      = np.zeros(n, dtype=float)
    buffer_J[0] = float(hw.buffer_init_J)
    is_active = (initial_state == "active")

    for k in range(n):
        E = buffer_J[k]
        if is_active and E <= enter:
            is_active = False
        elif not is_active and E >= exit_:
            is_active = True
        duty[k] = 1.0 if is_active else 0.0
        if k < n - 1:
            load_W = hw.p_active_W if is_active else hw.p_hibernate_W
            E_next = E + (solar_power_W[k] - load_W) * dt_s
            E_next = max(0.0, E_next)
            if E_next > cap and E_next >= exit_:
                E_next = cap
            buffer_J[k + 1] = E_next

    uptime_hours = float(np.sum(duty[:-1]) * dt_hours)
    return uptime_hours, duty, buffer_J


def simulate_thresholded_duty_strategy(
    solar_power_W,
    dt_hours: float,
    E_low_J: float,
    E_high_J: float,
    cap_J: float,
    E_init_J: float,
    P_awake_W: float,
    P_sleep_W: float,
    P_deep_W: float = 0.0,
    duty: float = 0.5,
    sleeps_per_hour: int = 4,
    start_phase_hours: float = 0.0,
):
    """
    Energy-blind duty-cycle strategy with hardware voltage thresholds.

    Device alternates ON/OFF at a fixed duty cycle regardless of buffer state.
    When the buffer drops below E_low_J it enters hard sleep until the buffer
    recovers to E_high_J.

    Returns:
        uptime_hours (float), duty_trace (0/1), buffer_J,
        mode_trace (0=hard sleep, 1=duty sleep, 2=awake)
    """
    solar = np.asarray(solar_power_W, dtype=float)
    n     = len(solar)
    dt_s  = float(dt_hours) * 3600.0
    cap   = float(cap_J)
    low   = float(np.clip(E_low_J,  0.0, cap))
    high  = float(np.clip(E_high_J, 0.0, cap))
    if high < low:
        low, high = high, low
    duty_frac    = float(np.clip(duty, 0.0, 1.0))
    cycle_len_hr = 1.0 / float(max(1e-6, sleeps_per_hour))
    on_len_hr    = duty_frac * cycle_len_hr

    buffer_J   = np.zeros(n, dtype=float)
    duty_trace = np.zeros(n, dtype=float)
    mode_trace = np.zeros(n, dtype=int)
    buffer_J[0] = float(np.clip(E_init_J, 0.0, cap))
    hard_latch  = False

    for k in range(n):
        E = buffer_J[k]
        if hard_latch:
            if E >= high:
                hard_latch = False
        else:
            if E <= low:
                hard_latch = True

        if hard_latch:
            is_on, P_load, mode = False, float(P_deep_W), 0
        else:
            phase = ((start_phase_hours + k * float(dt_hours)) % cycle_len_hr)
            is_on  = (phase < on_len_hr)
            P_load = float(P_awake_W) if is_on else float(P_sleep_W)
            mode   = 2 if is_on else 1

        duty_trace[k] = 1.0 if is_on else 0.0
        mode_trace[k] = mode
        if k < n - 1:
            E_next = E + (solar[k] - P_load) * dt_s
            buffer_J[k + 1] = float(np.clip(E_next, 0.0, cap))

    uptime_hours = float(np.sum(duty_trace[:-1]) * float(dt_hours))
    return uptime_hours, duty_trace, buffer_J, mode_trace


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _resolve_energy_blind_params(s, cap_J):
    """Resolve energy-blind strategy thresholds (absolute J or fractional)."""
    E_low_J  = float(s["E_low_J"])  if "E_low_J"  in s else float(s.get("E_low_frac",  0.10)) * float(cap_J)
    E_high_J = float(s["E_high_J"]) if "E_high_J" in s else float(s.get("E_high_frac", 0.60)) * float(cap_J)
    return {
        "E_low_J": E_low_J,
        "E_high_J": E_high_J,
        "duty": float(s.get("duty", 0.5)),
        "sleeps_per_hour": float(s.get("sleeps_per_hour", 4)),
        "start_phase_hours": float(s.get("start_phase_hours", 0.0)),
    }


def _resolve_hibernation_thresholds_J(strategy_name, strategy_params, cap_J, case_fixed_settings=None):
    """Return (enter_J, exit_J) in Joules from strategy params or pre-computed fixed settings."""
    if case_fixed_settings is not None:
        thresholds = case_fixed_settings.get("hib_thresholds_J", {}).get(strategy_name)
        if thresholds is not None:
            return float(thresholds["enter_J"]), float(thresholds["exit_J"])
    enter_J = float(strategy_params["enter_J"]) if "enter_J" in strategy_params \
              else float(strategy_params["enter_frac"]) * cap_J
    exit_J  = float(strategy_params["exit_J"])  if "exit_J"  in strategy_params \
              else float(strategy_params["exit_frac"])  * cap_J
    enter_J = max(enter_J, VOLTAGE_LOW_J)
    exit_J  = max(exit_J,  enter_J + 50.0)
    return float(enter_J), float(exit_J)


def sweep_thresholds_uptime_J(
    solar_power_W,
    hw: HardwareInputs,
    enter_J_values,
    exit_J_values,
    dt_hours: float = 1.0,
    min_hysteresis_J: float = 50.0,
):
    """
    Sweep all valid (enter, exit) threshold pairs and record uptime.

    Returns:
        uptime_grid: 2-D ndarray [len(enter_J_values) × len(exit_J_values)],
                     NaN for pairs where exit < enter + min_hysteresis_J
        rows: list of dicts with enter_J, exit_J, uptime_hours
    """
    enter_J_values = np.asarray(enter_J_values, dtype=float)
    exit_J_values  = np.asarray(exit_J_values,  dtype=float)
    uptime_grid = np.full((len(enter_J_values), len(exit_J_values)), np.nan)
    rows = []
    for i, enter_J in enumerate(enter_J_values):
        for j, exit_J in enumerate(exit_J_values):
            if exit_J < enter_J + float(min_hysteresis_J):
                continue
            uptime_h, _, _ = simulate_hibernation_strategy(
                solar_power_W=solar_power_W, hw=hw,
                enter_threshold_J=float(enter_J), exit_threshold_J=float(exit_J),
                initial_state="active", dt_hours=dt_hours,
            )
            uptime_grid[i, j] = uptime_h
            rows.append({"enter_J": float(enter_J), "exit_J": float(exit_J),
                         "uptime_hours": float(uptime_h)})
    return uptime_grid, rows


# ---------------------------------------------------------------------------
# Figure 1 — Buffer sizing procedure
# ---------------------------------------------------------------------------

def plot_buffer_sizing(
    E_size,
    cap_J,
    solar_power_W_min,
    start_date,
    location_name,
    dt_hours=1.0 / 60.0,
    show_plot=True,
    save_dir=None,
    filename="buffer_sizing",
):
    """
    Two-panel sizing visualization:
      top    — solar power trace
      bottom — accumulated net energy (floored at 0) with optimal buffer size marked.
    """
    FONT_SIZE   = 26
    FONT_WEIGHT = "bold"
    plt.rcParams.update({
        "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "font.weight": FONT_WEIGHT, "axes.labelweight": FONT_WEIGHT,
    })

    n     = len(E_size)
    time  = pd.date_range(start=start_date, periods=n, freq=pd.Timedelta(hours=float(dt_hours)))
    i_max = int(np.argmax(E_size))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax1.plot(time, np.asarray(solar_power_W_min[:n], dtype=float),
             color="#E5751F", linewidth=1.5)
    ax1.set_ylabel("Solar Power (W)", fontweight=FONT_WEIGHT)
    ax1.set_title(f"Energy Buffer Sizing — {location_name}", fontweight=FONT_WEIGHT)
    ax1.margins(x=0)

    ax2.plot(time, E_size, color="#861F41", linewidth=1.5, label="Accumulated net energy")
    ax2.axhline(cap_J, color="purple", linestyle="--", linewidth=2.0,
                label=f"Optimal buffer size: {cap_J / 1000:.1f} kJ")
    ax2.scatter([time[i_max]], [E_size[i_max]], color="purple", s=120, zorder=5)
    ax2.set_ylabel("Energy (J)", fontweight=FONT_WEIGHT)
    ax2.legend(fontsize=FONT_SIZE - 4)
    ax2.margins(x=0)
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax2.get_xticklabels(), rotation=0, ha="center")

    fig.tight_layout(h_pad=0.4)
    _save_or_show_figure(fig, show_plot=show_plot, save_dir=save_dir, filename=filename)


# ---------------------------------------------------------------------------
# Figure 2 — Uptime heatmap
# ---------------------------------------------------------------------------

def plot_uptime_heatmap(
    uptime_grid,
    enter_J_values,
    exit_J_values,
    title,
    show_plot=True,
    save_dir=None,
    filename="heatmap",
):
    """Heatmap of uptime (hours) over all valid (enter, exit) threshold pairs."""
    FONT_SIZE   = 26
    FONT_WEIGHT = "bold"

    data        = np.array(uptime_grid, dtype=float)
    data_masked = np.ma.masked_invalid(data)
    extent = [float(exit_J_values[0]), float(exit_J_values[-1]),
              float(enter_J_values[0]), float(enter_J_values[-1])]

    plt.figure(figsize=(14, 11))
    im = plt.imshow(data_masked, origin="lower", aspect="auto", extent=extent,
                    interpolation="nearest",
                    vmin=float(np.nanmin(data)), vmax=float(np.nanmax(data)))

    cbar = plt.colorbar(im)
    cbar.set_label("Uptime (hours)", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight(FONT_WEIGHT)

    ax = plt.gca()
    ax.set_xlabel("Exit hibernation threshold (Joules)",  fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.set_ylabel("Enter hibernation threshold (Joules)", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.set_title(title, fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(FONT_WEIGHT)

    if np.any(~np.isnan(data)):
        for fn, label, ha, va in [(np.nanargmax, "MAX", "right", "top"),
                                   (np.nanargmin, "MIN", "left",  "bottom")]:
            ir, ic = np.unravel_index(fn(data), data.shape)
            val = data[ir, ic]
            ax.scatter(exit_J_values[ic], enter_J_values[ir], s=1000, marker="*", zorder=5)
            pad = " " if ha == "left" else ""
            ax.text(exit_J_values[ic], enter_J_values[ir],
                    f"{pad}{label}\n{pad}{val:.1f} h",
                    ha=ha, va=va, fontsize=FONT_SIZE, fontweight=FONT_WEIGHT, zorder=10)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(left=0.12, right=1.0)
    _save_or_show_figure(plt.gcf(), show_plot=show_plot, save_dir=save_dir, filename=filename)


# ---------------------------------------------------------------------------
# Figure 3 — Stacked per-strategy energy traces
# ---------------------------------------------------------------------------

def plot_stacked_strategies_per_location(
    strategy_results,
    location_name,
    start_date,
    end_date,
    dt_hours,
    voltage_low_J,
    voltage_high_J,
    buffer_capacity_J=None,
    max_uptime_h=None,
    show_plot=True,
    save_dir=None,
    filename=None,
):
    """
    Stacked N-panel plot — one subplot per strategy, sharing the x-axis.

    Each entry in strategy_results must be a dict with keys:
        name, buffer_J, duty_trace, mode_trace,
        E_low_J, E_high_J, show_strategy_thresholds
    """
    n_strat     = len(strategy_results)
    FONT_SIZE   = 30
    FONT_WEIGHT = "bold"

    fig, axes = plt.subplots(n_strat, 1, sharex=True, figsize=(24, 4 * n_strat + 2))
    if n_strat == 1:
        axes = [axes]

    def _color(s):
        if s == 2: return "tab:blue",   0.26
        if s == 1: return "tab:orange", 0.30
        return "tab:red", 0.22

    for idx, (ax, res) in enumerate(zip(axes, strategy_results)):
        E    = np.asarray(res["buffer_J"],   dtype=float)
        duty = np.asarray(res["duty_trace"], dtype=float)
        mode = np.asarray(res["mode_trace"], dtype=int)
        n    = len(E)
        time = pd.date_range(start=start_date, periods=n,
                             freq=pd.Timedelta(hours=float(dt_hours)))

        # Map simulation mode → display state (0=off, 1=hib, 2=on)
        state = np.zeros(n, dtype=int)
        state[mode == 0] = 0
        state[(mode == 1) & (duty <= 0.5)] = 1
        state[(mode == 2) | ((mode == 1) & (duty > 0.5))] = 2

        si, cur = 0, state[0]
        for i in range(1, n):
            if state[i] != cur:
                c, a = _color(cur)
                ax.axvspan(time[si], time[i], color=c, alpha=a, lw=0)
                si, cur = i, state[i]
        if n > 1:
            c, a = _color(cur)
            ax.axvspan(time[si], time[-1], color=c, alpha=a, lw=0)

        ax.plot(time, E, color="black", linewidth=2.5)

        if res.get("show_strategy_thresholds", True):
            ax.axhline(res["E_low_J"],  color="C0", linestyle="--",          linewidth=2.5)
            ax.axhline(res["E_high_J"], color="C0", linestyle=(0,(3,1,1,1)), linewidth=2.5)
        if buffer_capacity_J is not None:
            ax.axhline(float(buffer_capacity_J), color="purple", linestyle="--", linewidth=2.5)
        if voltage_high_J is not None:
            ax.axhline(float(voltage_high_J), color="red", linestyle="--", linewidth=2.5)
        if voltage_low_J is not None:
            ax.axhline(float(voltage_low_J),  color="red", linestyle=":",  linewidth=2.5)

        uptime_h = float(np.sum(duty[:-1] > 0.5) * float(dt_hours))
        letter   = f"{chr(ord('a') + idx)}.) "
        if max_uptime_h is not None:
            pct   = 100 * uptime_h / max_uptime_h
            title = f"{letter}{res['name']}  —  Uptime: {uptime_h:.1f} h / {max_uptime_h:.1f} h  ({pct:.0f}%)"
        else:
            title = f"{letter}{res['name']}  —  Uptime: {uptime_h:.1f} h"

        ax.set_title(title, fontsize=26, fontweight=FONT_WEIGHT)
        ax.set_ylabel("Energy (J)", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
        ax.margins(x=0)
        for tick in ax.get_yticklabels():
            tick.set_fontweight(FONT_WEIGHT)
        if idx < n_strat - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    y_min = min(ax.get_ylim()[0] for ax in axes)
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(axes[-1].get_xticklabels(), rotation=0, ha="center", fontsize=FONT_SIZE)
    for tick in axes[-1].get_xticklabels():
        tick.set_fontweight(FONT_WEIGHT)
    axes[-1].set_xlabel("", fontsize=FONT_SIZE)

    fig.tight_layout(rect=[0, 0.01, 1, 0.99], h_pad=0.4)
    _save_or_show_figure(fig, show_plot=show_plot, save_dir=save_dir, filename=filename)


# ---------------------------------------------------------------------------
# Figure 4 — Uptime % of maximum across all environments
# ---------------------------------------------------------------------------

def run_uptime_percent_bar_chart(
    site_template,
    hw_template,
    show_plot=True,
    save_dir=None,
    filename="uptime_percent_of_maximum",
    solar_mode="dni",
):
    """
    Clustered bar chart: each strategy's uptime as % of theoretical maximum
    across all five deployment environments defined in GRID_CASES.

    Returns:
        pct_mat [n_cases × n_strategies], uptime_mat, optimal_uptime_h [n_cases]
    """
    sizing_strategy = {
        "mode": "constant", "signal": "solar",
        "p_hibernate_W": 1e-9, "on_threshold_W": 0.30,
        "off_threshold_W": 0.10, "initial_state": "active",
    }

    case_names  = [c["name"] for c in GRID_CASES]
    strat_names = list(STRATEGIES.keys())
    P_device_W  = float(hw_template.p_active_W)
    dt_h        = 1.0 / 60.0   # 1-minute resolution

    n_cases, n_strats = len(GRID_CASES), len(STRATEGIES)
    pct_mat = np.zeros((n_cases, n_strats))
    up_mat  = np.zeros((n_cases, n_strats))
    opt_h   = np.zeros(n_cases)

    for i, case in enumerate(GRID_CASES):
        site = replace(site_template,
                       latitude=case["lat"], longitude=case["lon"],
                       start_date=case["start"], end_date=case["end"],
                       location=case["name"])

        cap_J, cap_Wh, _, solar_W_min, *_ = compute_required_buffer_max_accumulated_floor0(
            latitude=site.latitude, longitude=site.longitude,
            start_date=site.start_date, end_date=site.end_date,
            location=site.location, active_area_m2=hw_template.active_area_m2,
            efficiency=hw_template.efficiency, max_power_draw_W=hw_template.p_active_W,
            strategy=sizing_strategy, dt_minutes=1, e0_J=2000, e_floor_J=1500,
            dni_threshold=5.0, solar_mode=solar_mode, rock_amplitude_deg=0.0,
        )
        cap_J = float(cap_J)
        total_solar_J = float(np.sum(np.asarray(solar_W_min[:-1], dtype=float)) * dt_h * 3600.0)
        optimal_h     = (total_solar_J + hw_template.buffer_init_J) / P_device_W / 3600.0
        opt_h[i]      = optimal_h

        hw = replace(hw_template, buffer_capacity_J=cap_J,
                     buffer_init_J=min(float(hw_template.buffer_init_J), cap_J))

        for j, sname in enumerate(strat_names):
            s = STRATEGIES[sname]
            if s["type"] == "hib":
                en_J, ex_J = _resolve_hibernation_thresholds_J(sname, s, cap_J)
                uptime_h, _, _ = simulate_hibernation_strategy(
                    solar_power_W=solar_W_min, hw=hw,
                    enter_threshold_J=en_J, exit_threshold_J=ex_J,
                    initial_state="active", dt_hours=dt_h,
                )
            elif s["type"] == "energy_blind":
                eb = _resolve_energy_blind_params(s, cap_J)
                uptime_h, _, _, _ = simulate_thresholded_duty_strategy(
                    solar_power_W=solar_W_min, dt_hours=dt_h,
                    E_low_J=VOLTAGE_LOW_J, E_high_J=VOLTAGE_HIGH_J,
                    cap_J=cap_J, E_init_J=float(hw.buffer_init_J),
                    P_awake_W=float(hw.p_active_W), P_sleep_W=float(hw.p_hibernate_W),
                    P_deep_W=float(hw.p_hibernate_W), duty=eb["duty"],
                    sleeps_per_hour=eb["sleeps_per_hour"],
                    start_phase_hours=eb["start_phase_hours"],
                )
            else:
                raise ValueError(f"Unknown strategy type: {s['type']}")

            up_mat[i, j]  = float(uptime_h)
            pct_mat[i, j] = 100.0 * float(uptime_h) / optimal_h if optimal_h > 0 else float("nan")

        print(f"  {case['name']}: buffer={cap_J:.0f} J ({cap_Wh:.3f} Wh), "
              f"max_uptime={optimal_h:.1f} h")

    # --- Plot ---
    FONT_SIZE   = 26
    FONT_WEIGHT = "bold"
    VT_COLORS   = ["#861F41", "#E5751F", "#006272", "#6A7F3C", "#E9B44C", "#75787B"]

    x       = np.arange(n_cases)
    bar_w   = 0.97 / n_strats
    offsets = (np.arange(n_strats) - (n_strats - 1) / 2.0) * bar_w

    fig, ax = plt.subplots(figsize=(16, 9.58), dpi=100)
    ax.axhline(100.0, color="red", linestyle="--", linewidth=2.5, label="100% optimal")

    for j, sname in enumerate(strat_names):
        bars = ax.bar(x + offsets[j], pct_mat[:, j], width=bar_w,
                      label=sname, color=VT_COLORS[j % len(VT_COLORS)])
        for b in bars:
            h = float(b.get_height())
            if np.isfinite(h):
                ax.text(b.get_x() + b.get_width() / 2.0, h + 0.3,
                        f"{round(h):.0f}", ha="center", va="bottom",
                        fontsize=FONT_SIZE, fontweight=FONT_WEIGHT, clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(case_names, rotation=0, ha="center",
                       fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.set_xlim(-0.5, n_cases - 0.5)
    ax.tick_params(axis="x", pad=12)
    for k, tick in enumerate(ax.xaxis.get_major_ticks()):
        if k % 2 == 1:
            tick.set_pad(52)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(0, 104)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    for tick in ax.get_yticklabels():
        tick.set_fontweight(FONT_WEIGHT)
    ax.set_ylabel("Uptime (% of Maximum Uptime)", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.set_title("Uptime Percentage of Maximum by Strategy and Environment",
                 fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    legend = ax.legend(fontsize=FONT_SIZE - 2, loc="lower right")
    for text in legend.get_texts():
        text.set_fontweight(FONT_WEIGHT)
    fig.tight_layout()
    plt.subplots_adjust(left=0.10, bottom=0.14, top=0.92, right=0.98)
    _save_or_show_figure(fig, show_plot=show_plot, save_dir=save_dir, filename=filename, dpi=100)

    return pct_mat, up_mat, opt_h


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # ── Hardware + site configuration ──────────────────────────────────────────
    site = SiteInputs(
        latitude=seattle_lat,
        longitude=seattle_long,
        start_date="2025-06-09",
        end_date="2025-06-22",
        location="Intermittent Energy",
    )
    hw = HardwareInputs(
        active_area_m2=shore_pod_realistic_SA_m2,
        efficiency=sb_sca_efficiency,
        p_active_W=0.25,
        p_hibernate_W=0.0,
        buffer_capacity_J=13511.6,
        buffer_init_J=2000.0,
    )

    # Shared sizing call — minute-resolution solar trace + buffer size for
    # the Intermittent Energy case; results reused by Figures 1–3.
    print("Fetching irradiance and computing buffer size...")
    _sizing_strategy = {
        "mode": "constant", "signal": "solar",
        "p_hibernate_W": 1e-9, "on_threshold_W": 0.30,
        "off_threshold_W": 0.10, "initial_state": "active",
    }
    cap_J, _, _, solar_W_min, _, _, E_size, *_ = compute_required_buffer_max_accumulated_floor0(
        latitude=site.latitude, longitude=site.longitude,
        start_date=site.start_date, end_date=site.end_date,
        location=site.location, active_area_m2=hw.active_area_m2,
        efficiency=hw.efficiency, max_power_draw_W=hw.p_active_W,
        strategy=_sizing_strategy, dt_minutes=1, e0_J=2000, e_floor_J=1500,
        dni_threshold=5.0, solar_mode="dni", rock_amplitude_deg=0.0,
    )
    cap_J  = float(cap_J)
    dt_h   = 1.0 / 60.0   # 1 minute in hours
    hw_sim = replace(hw, buffer_capacity_J=cap_J,
                     buffer_init_J=min(hw.buffer_init_J, cap_J))
    print(f"  Optimal buffer capacity: {cap_J:.0f} J ({cap_J/3600:.3f} Wh)\n")

    # ── Figure 1 — Buffer sizing ───────────────────────────────────────────────
    print("Plotting buffer sizing...")
    plot_buffer_sizing(
        E_size=E_size, cap_J=cap_J, solar_power_W_min=solar_W_min,
        start_date=site.start_date, location_name=site.location,
        dt_hours=dt_h, show_plot=True,
    )

    # ── Figure 2 — Heatmap ────────────────────────────────────────────────────
    # Use hourly resolution for speed (101×101 = 10 201 sims; ~5 050 valid).
    print("Running threshold sweep for heatmap (hourly resolution, ~1 min)...")
    _, solar_W_hourly, _ = compute_hourly_solar_power_W(site, hw_sim)
    hw_hourly = replace(hw_sim, buffer_init_J=min(hw.buffer_init_J, cap_J))
    enter_vals = np.linspace(cap_J * 0.05, cap_J * 0.95, 101)
    exit_vals  = np.linspace(cap_J * 0.05, cap_J * 0.95, 101)
    uptime_grid, _ = sweep_thresholds_uptime_J(
        solar_power_W=solar_W_hourly, hw=hw_hourly,
        enter_J_values=enter_vals, exit_J_values=exit_vals,
        dt_hours=1.0, min_hysteresis_J=50.0,
    )
    plot_uptime_heatmap(
        uptime_grid=uptime_grid, enter_J_values=enter_vals, exit_J_values=exit_vals,
        title=f"Uptime Heatmap — {site.location}",
        show_plot=True,
    )

    # ── Figure 3 — Stacked strategy traces ────────────────────────────────────
    print("Simulating strategy traces...")
    total_solar_J = float(np.sum(solar_W_min)) * dt_h * 3600.0
    max_uptime_h  = min(
        len(solar_W_min) * dt_h,
        (total_solar_J + hw_sim.buffer_init_J) / hw.p_active_W / 3600.0,
    )
    print(f"  Max theoretical uptime: {max_uptime_h:.1f} h  "
          f"({100 * max_uptime_h / (len(solar_W_min) * dt_h):.1f}% of span)")

    stacked = []
    for sname, s in STRATEGIES.items():
        if s["type"] == "hib":
            en_J, ex_J = _resolve_hibernation_thresholds_J(sname, s, cap_J)
            _, duty, E_buf = simulate_hibernation_strategy(
                solar_power_W=solar_W_min, hw=hw_sim,
                enter_threshold_J=en_J, exit_threshold_J=ex_J,
                initial_state="active", dt_hours=dt_h,
            )
            mode = np.where(np.asarray(duty) > 0.5, 2, 1).astype(int)
            stacked.append({"name": sname, "buffer_J": E_buf, "duty_trace": duty,
                            "mode_trace": mode, "E_low_J": en_J, "E_high_J": ex_J,
                            "show_strategy_thresholds": True})
        elif s["type"] == "energy_blind":
            eb = _resolve_energy_blind_params(s, cap_J)
            _, duty, E_buf, mode = simulate_thresholded_duty_strategy(
                solar_power_W=solar_W_min, dt_hours=dt_h,
                E_low_J=VOLTAGE_LOW_J, E_high_J=VOLTAGE_HIGH_J,
                cap_J=cap_J, E_init_J=float(hw_sim.buffer_init_J),
                P_awake_W=float(hw.p_active_W), P_sleep_W=float(hw.p_hibernate_W),
                P_deep_W=float(hw.p_hibernate_W), duty=eb["duty"],
                sleeps_per_hour=eb["sleeps_per_hour"],
                start_phase_hours=eb["start_phase_hours"],
            )
            stacked.append({"name": sname, "buffer_J": E_buf, "duty_trace": duty,
                            "mode_trace": mode, "E_low_J": VOLTAGE_LOW_J,
                            "E_high_J": VOLTAGE_HIGH_J, "show_strategy_thresholds": False})

    plot_stacked_strategies_per_location(
        strategy_results=stacked, location_name=site.location,
        start_date=site.start_date, end_date=site.end_date,
        dt_hours=dt_h, voltage_low_J=VOLTAGE_LOW_J, voltage_high_J=VOLTAGE_HIGH_J,
        buffer_capacity_J=cap_J, max_uptime_h=max_uptime_h,
        show_plot=True,
    )

    # ── Figure 4 — Uptime % of maximum (all environments) ────────────────────
    print("\nRunning all-environment comparison (takes a few minutes)...")
    run_uptime_percent_bar_chart(
        site_template=site, hw_template=hw,
        show_plot=True,
        solar_mode="dni",
    )


if __name__ == "__main__":
    main()
