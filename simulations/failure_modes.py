"""
Three energy buffer failure modes — Seattle, June 9–10 2025.

  1. Overflow          : buffer too small, incoming solar energy wasted at cap
  2. Depletion         : buffer drains to 0, high turn-on threshold = long dead period
  3. Unnecessary hib   : thresholds too conservative, device sleeps with usable energy

All three use the same 2-day solar trace. Uptime is reported relative to the
theoretical maximum (total harvested energy / power draw).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import seattle_lat, seattle_long, shore_pod_realistic_SA_m2, sb_sca_efficiency
from irradiance import get_irradiance_data


def plot_failure_modes(show_plot=True, save_dir=None, filename="failure_modes"):
    # ── Simulation parameters ──────────────────────────────────────────────────
    LAT          = seattle_lat
    LON          = seattle_long
    START_DATE   = "2025-06-09"
    END_DATE     = "2025-06-10"
    LOCATION     = "Seattle"
    ACTIVE_AREA  = shore_pod_realistic_SA_m2
    EFFICIENCY   = sb_sca_efficiency
    POWER_DRAW_W = 0.4
    DT_MIN       = 1
    DNI_THRESH   = 1.0

    # ── 1-minute solar power trace ─────────────────────────────────────────────
    hourly_df, _, hourly_dni, _ = get_irradiance_data(LAT, LON, START_DATE, END_DATE, LOCATION)

    t_hourly   = pd.to_datetime(hourly_df["date"], utc=True).dt.tz_localize(None)
    dni_hourly = np.asarray(hourly_dni, dtype=float)

    t_min = pd.date_range(start=t_hourly.iloc[0], end=t_hourly.iloc[-1], freq=f"{DT_MIN}min")
    _ref          = t_hourly.iloc[0]
    _t_hourly_sec = (t_hourly - _ref).dt.total_seconds().values
    _t_min_sec    = (t_min    - _ref).total_seconds().values
    dni_min = np.interp(_t_min_sec, _t_hourly_sec, dni_hourly)
    dni_min[dni_min < DNI_THRESH] = 0.0

    times     = t_min.to_numpy()
    P_harvest = dni_min * ACTIVE_AREA * EFFICIENCY
    dt_sec    = DT_MIN * 60.0
    n         = len(times)

    INITIAL_E_J      = 5000.0
    total_solar_J    = float(np.sum(P_harvest) * dt_sec)
    sim_duration_h   = n * dt_sec / 3600.0
    energy_ceiling_h = (total_solar_J + INITIAL_E_J) / POWER_DRAW_W / 3600.0
    max_uptime_h     = min(sim_duration_h, energy_ceiling_h)

    print(f"Total solar harvested : {total_solar_J:.0f} J  ({total_solar_J/3600:.2f} Wh)")
    print(f"Max theoretical uptime: {max_uptime_h:.2f} h")

    def uptime_h(on_mask):
        return float(np.sum(on_mask)) * dt_sec / 3600.0

    # ── Failure Mode 1: Overflow ───────────────────────────────────────────────
    LOW_V_J      = 1000.0
    CAP1_J       = 10000.0
    ON_THRESH1_J =  8000.0

    buf1 = np.zeros(n); on1 = np.zeros(n, dtype=bool); overflow1 = np.zeros(n)
    E = INITIAL_E_J; device_on = True
    for k in range(n):
        buf1[k] = E; on1[k] = device_on
        P_load = POWER_DRAW_W if device_on else 0.0
        E_raw  = E + (P_harvest[k] - P_load) * dt_sec
        overflow1[k] = max(0.0, E_raw - CAP1_J)
        E = max(0.0, min(CAP1_J, E_raw))
        if     device_on and E <= LOW_V_J:        device_on = False
        elif not device_on and E >= ON_THRESH1_J: device_on = True
    ut1 = uptime_h(on1)
    total_overflow_J = float(np.sum(overflow1))

    # ── Failure Mode 2: Depletion ──────────────────────────────────────────────
    CAP2_J       = 12000.0
    ON_THRESH2_J =  8000.0

    buf2 = np.zeros(n); on2 = np.zeros(n, dtype=bool)
    E = INITIAL_E_J; device_on = True
    for k in range(n):
        buf2[k] = E; on2[k] = device_on
        P_load = POWER_DRAW_W if device_on else 0.0
        E = max(0.0, min(CAP2_J, E + (P_harvest[k] - P_load) * dt_sec))
        if     device_on and E <= LOW_V_J:        device_on = False
        elif not device_on and E >= ON_THRESH2_J: device_on = True
    ut2 = uptime_h(on2)

    # ── Failure Mode 3: Unnecessary hibernation ────────────────────────────────
    CAP3_J      = 12000.0
    HIB_ENTER_J =  4000.0
    HIB_EXIT_J  =  7000.0

    buf3 = np.zeros(n); on3 = np.zeros(n, dtype=bool); hib3 = np.zeros(n, dtype=bool)
    E = INITIAL_E_J; state = "full"
    for k in range(n):
        buf3[k] = E; on3[k] = (state == "full"); hib3[k] = (state == "hib")
        P_load = POWER_DRAW_W if state == "full" else 0.0
        E = max(0.0, min(CAP3_J, E + (P_harvest[k] - P_load) * dt_sec))
        if state == "full" and E <= HIB_ENTER_J: state = "hib"
        elif state == "hib" and E >= HIB_EXIT_J: state = "full"
    ut3 = uptime_h(on3)

    # ── Plotting ───────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.size": 26, "axes.titlesize": 26, "axes.labelsize": 26,
        "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 26,
        "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold",
    })

    COLOR_ON       = "#aec6e8"
    COLOR_DEPLETED = "#f4a0a0"
    COLOR_HIB      = "#f4c07a"
    t_plot = pd.to_datetime(times)

    def shade_regions(ax, t, regions):
        t = np.asarray(t, dtype="datetime64[ns]")
        dt = t[1] - t[0]
        for mask, color, label in regions:
            mask = np.asarray(mask, dtype=bool)
            labeled = False
            starts = np.where(mask & ~np.r_[False, mask[:-1]])[0]
            ends   = np.where(mask & ~np.r_[mask[1:],  False])[0]
            for s, e in zip(starts, ends):
                x0 = pd.Timestamp(t[s]); x1 = pd.Timestamp(t[e] + dt)
                ax.axvspan(x0, x1, facecolor=color, alpha=0.45,
                           label=label if not labeled else "_", zorder=0)
                labeled = True

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle("Energy Buffer Failure Modes — Seattle, June 9–10 2025", fontsize=22, y=0.995)

    ax = axes[0]
    shade_regions(ax, times, [(on1, COLOR_ON, "On"), (~on1, COLOR_DEPLETED, "Off")])
    ax.plot(t_plot, buf1, color="tab:blue", linewidth=4, label="_")
    ax.axhline(CAP1_J,       color="purple", linestyle="--", linewidth=1.4, label="Buffer Capacity")
    ax.axhline(ON_THRESH1_J, color="red",    linestyle="--", linewidth=1.4, label="High-Voltage Threshold")
    ax.axhline(LOW_V_J,      color="red",    linestyle=":",  linewidth=1.8, label="Low-Voltage Threshold")
    ax.set_ylabel("Energy [J]"); ax.set_ylim(0, 15000)
    ax.set_title(f"Buffer Overflow  —  Uptime: {ut1:.1f} h / {max_uptime_h:.1f} h  ({100*ut1/max_uptime_h:.0f}%)", loc="left")

    ax = axes[1]
    shade_regions(ax, times, [(on2, COLOR_ON, "On"), (~on2, COLOR_DEPLETED, "Off")])
    ax.plot(t_plot, buf2, color="tab:orange", linewidth=4, label="_")
    ax.axhline(CAP2_J,       color="purple", linestyle="--", linewidth=1.4, label="Buffer Capacity")
    ax.axhline(ON_THRESH2_J, color="red",    linestyle="--", linewidth=1.4, label="High-Voltage Threshold")
    ax.axhline(LOW_V_J,      color="red",    linestyle=":",  linewidth=1.8, label="Low-Voltage Threshold")
    ax.set_ylabel("Energy [J]"); ax.set_ylim(0, 15000)
    ax.set_title(f"Buffer Depletion  —  Uptime: {ut2:.1f} h / {max_uptime_h:.1f} h  ({100*ut2/max_uptime_h:.0f}%)", loc="left")

    ax = axes[2]
    shade_regions(ax, times, [(on3, COLOR_ON, "On"), (hib3, COLOR_HIB, "Hibernation")])
    ax.plot(t_plot, buf3, color="darkgreen", linewidth=4, label="_")
    ax.axhline(CAP3_J,      color="purple",   linestyle="--", linewidth=1.4, label="Buffer Capacity")
    ax.axhline(HIB_ENTER_J, color="tab:blue", linestyle="--", linewidth=1.4, label="Hibernation Threshold")
    ax.axhline(HIB_EXIT_J,  color="tab:blue", linestyle="--", linewidth=1.4, label="_")
    ax.axhline(LOW_V_J,     color="red",      linestyle=":",  linewidth=1.8, label="Low-Voltage Threshold")
    ax.set_ylabel("Energy [J]"); ax.set_ylim(0, 15000)
    ax.set_title(f"Unnecessary Hibernation  —  Uptime: {ut3:.1f} h / {max_uptime_h:.1f} h  ({100*ut3/max_uptime_h:.0f}%)", loc="left")

    def _fmt_time(x, pos=None):
        dt = mdates.num2date(x)
        h = dt.hour % 12 or 12
        return f"{h}:00 {'AM' if dt.hour < 12 else 'PM'}"

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_time))
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.margins(x=0)
    axes[-1].set_xlabel("")
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=0, ha="center")

    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h); all_labels.extend(l)
    seen, unique_h, unique_l = set(), [], []
    for h, l in zip(all_handles, all_labels):
        if l not in seen and not l.startswith("_"):
            seen.add(l); unique_h.append(h); unique_l.append(l)
    fig.legend(unique_h, unique_l, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=4, framealpha=0.95, fontsize=20)

    plt.tight_layout(rect=[0, 0.13, 1, 0.96], pad=0.4, h_pad=0.4)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
        fig.savefig(os.path.join(save_dir, safe + ".png"), dpi=300, bbox_inches="tight")
        print(f"Saved: {os.path.join(save_dir, safe + '.png')}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f"\nUptime summary (max theoretical = {max_uptime_h:.2f} h):")
    print(f"  Overflow sim        : {ut1:.2f} h  ({100*ut1/max_uptime_h:.1f}% of max)"
          f"   |  wasted overflow: {total_overflow_J:.0f} J ({total_overflow_J/3600:.2f} Wh)")
    print(f"  Depletion sim       : {ut2:.2f} h  ({100*ut2/max_uptime_h:.1f}% of max)")
    print(f"  Unnecessary hib sim : {ut3:.2f} h  ({100*ut3/max_uptime_h:.1f}% of max)")


if __name__ == "__main__":
    OCEANS_FIG_DIR = r"C:\Users\amber\Documents\VT\Shore Pod\OCEANS\OCEANS-2026-Paper\figures"
    plot_failure_modes(show_plot=True, save_dir=OCEANS_FIG_DIR, filename="pitfalls")
