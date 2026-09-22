"""
buffer_sizing.py — Minute-resolution energy buffer capacity sizing.

Sizing metric: required buffer = max(E(t)), where the energy trace E is
integrated from a minute-resolution solar power trace with a configurable
lower floor (E_floor) and initial condition (E0).

Main entry point: compute_required_buffer_max_accumulated_floor0()
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from config import seattle_lat, seattle_long, shore_pod_realistic_SA_m2, sb_sca_efficiency
from irradiance import get_minute_irradiance, get_minute_irradiance_scaled


@dataclass
class Inputs:
    """Convenience dataclass for sizing inputs."""
    latitude:        float
    longitude:       float
    active_area_m2:  float
    efficiency:      float
    max_power_draw_W: float
    start_date:      str
    end_date:        str
    location:        str = "site"
    dt_minutes:      int = 1


def compute_solar_power_W(
    latitude, longitude, start_date, end_date, location,
    active_area_m2, efficiency, dt_minutes=1,
    dni_threshold=5.0, solar_mode="dni", rock_amplitude_deg=30.0,
):
    """
    Return minute-resolution solar electrical power (W).

    solar_mode
    ----------
    "dni"        -- raw DNI (sun-tracking / normal-to-panel)
    "scaled"     -- DNI * cos(zenith) = direct_radiation (upright panel)
    "worst_wave" -- DNI * max(0, cos(zenith + rock_amplitude)) (worst-case tilt)
    "mean_wave"  -- cycle-averaged rocking loss

    Returns
    -------
    time_min     : pd.DatetimeIndex
    solar_power_W: np.ndarray
    irr_W_m2     : np.ndarray  (irradiance used before area/efficiency scaling)
    """
    if solar_mode == "dni":
        time_min, irr_min = get_minute_irradiance(
            latitude=latitude, longitude=longitude,
            start_date=start_date, end_date=end_date, location=location,
            dt_minutes=int(dt_minutes), dni_threshold=float(dni_threshold),
        )
    else:
        time_min, irr_min = get_minute_irradiance_scaled(
            latitude=latitude, longitude=longitude,
            start_date=start_date, end_date=end_date, location=location,
            dt_minutes=int(dt_minutes), dni_threshold=float(dni_threshold),
            mode=solar_mode, rock_amplitude_deg=float(rock_amplitude_deg),
        )

    solar_power_W = np.asarray(irr_min, dtype=float) * float(active_area_m2) * float(efficiency)
    return time_min, solar_power_W, np.asarray(irr_min, dtype=float)


def load_trace_from_strategy(solar_power_W, max_power_draw_W, strategy):
    """
    Build a minute-resolution load power trace from a strategy dict.

    Supported strategy modes
    ------------------------
    "constant"    -- device always on at max_power_draw_W
    "hibernation" -- hysteresis on signal (solar or net_no_buffer)
    """
    solar_power_W = np.asarray(solar_power_W, dtype=float)
    n = len(solar_power_W)

    mode      = strategy.get("mode", "constant")
    sig_kind  = strategy.get("signal", "solar")
    p_hib_W   = float(strategy.get("p_hibernate_W", 0.0))
    on_th     = float(strategy.get("on_threshold_W", 0.0))
    off_th    = float(strategy.get("off_threshold_W", 0.0))

    if off_th > on_th:
        off_th, on_th = on_th, off_th

    is_active = strategy.get("initial_state", "active") == "active"
    signal_W  = (
        solar_power_W - float(max_power_draw_W)
        if sig_kind == "net_no_buffer" else solar_power_W
    )

    load_power_W = np.zeros(n, dtype=float)
    duty         = np.zeros(n, dtype=float)

    for k in range(n):
        if mode == "hibernation":
            s = signal_W[k]
            if is_active and s <= off_th:
                is_active = False
            elif (not is_active) and s >= on_th:
                is_active = True
            load_power_W[k] = float(max_power_draw_W) if is_active else p_hib_W
            duty[k]         = 1.0 if is_active else 0.0
        else:
            load_power_W[k] = float(max_power_draw_W)
            duty[k]         = 1.0

    return load_power_W, duty, signal_W


def compute_energy_trace_floor0_J(net_power_W, dt_minutes, e0_J=0.0, e_floor_J=0.0):
    """
    Integrate net power to obtain an energy trace with a configurable floor.

        E[0] = max(E_floor, E0)
        E[k] = max(E_floor, E[k-1] + P_net[k] * dt)

    Returns np.ndarray of energy values (J).
    """
    net  = np.asarray(net_power_W, dtype=float)
    n    = len(net)
    if n == 0:
        return np.array([])

    dt_s    = float(dt_minutes) * 60.0
    floor_J = float(e_floor_J)
    E_J     = np.zeros(n, dtype=float)
    E_J[0]  = max(floor_J, float(e0_J))

    for k in range(1, n):
        E_J[k] = E_J[k - 1] + net[k] * dt_s
        if E_J[k] < floor_J:
            E_J[k] = floor_J

    return E_J


def size_buffer_by_max_accumulated_energy(E_J):
    """Return (cap_J, cap_Wh, idx_max) — the peak energy and its index."""
    E = np.asarray(E_J, dtype=float)
    if E.size == 0:
        return 0.0, 0.0, None
    idx_max = int(np.argmax(E))
    cap_J   = float(E[idx_max])
    cap_Wh  = cap_J / 3600.0
    return cap_J, cap_Wh, idx_max


def compute_required_buffer_max_accumulated_floor0(
    latitude, longitude, start_date, end_date, location,
    active_area_m2, efficiency, max_power_draw_W, strategy,
    dt_minutes=1, e0_J=0.0, e_floor_J=2000.0,
    dni_threshold=5.0, solar_mode="dni", rock_amplitude_deg=30.0,
):
    """
    Full sizing pipeline: fetch solar data, build load trace, integrate, size.

    Returns
    -------
    cap_J, cap_Wh, time, solar_power_W, load_power_W,
    net_power_W, E_J, idx_max, duty, signal_W, dni_W_m2
    """
    time, solar_power_W, dni_W_m2 = compute_solar_power_W(
        latitude=latitude, longitude=longitude,
        start_date=start_date, end_date=end_date, location=location,
        active_area_m2=active_area_m2, efficiency=efficiency,
        dt_minutes=dt_minutes, dni_threshold=dni_threshold,
        solar_mode=solar_mode, rock_amplitude_deg=rock_amplitude_deg,
    )

    load_power_W, duty, signal_W = load_trace_from_strategy(
        solar_power_W=solar_power_W,
        max_power_draw_W=max_power_draw_W,
        strategy=strategy,
    )

    net_power_W = solar_power_W - load_power_W
    E_J = compute_energy_trace_floor0_J(
        net_power_W=net_power_W,
        dt_minutes=dt_minutes,
        e0_J=e0_J,
        e_floor_J=e_floor_J,
    )

    cap_J, cap_Wh, idx_max = size_buffer_by_max_accumulated_energy(E_J)

    return (
        cap_J, cap_Wh, time, solar_power_W, load_power_W,
        net_power_W, E_J, idx_max, duty, signal_W, dni_W_m2,
    )


# ── Diagnostic plots (optional) ───────────────────────────────────────────────

def plot_traces(time, solar_power_W, load_power_W, net_power_W):
    """Quick diagnostic: net power and solar/load overlay."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(time, net_power_W)
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_ylabel("Net Power (W)")
    axes[0].set_title("Net Power Trace (Solar - Load)")

    axes[1].plot(time, solar_power_W, label="Solar (W)")
    axes[1].plot(time, load_power_W,  label="Load (W)")
    axes[1].set_ylabel("Power (W)")
    axes[1].legend()
    axes[1].set_title("Solar / Load — Minute Sampling")

    plt.tight_layout()
    plt.show()


def plot_energy_trace_with_max(time, E_J, idx_max, cap_J):
    """Quick diagnostic: energy trace with peak marker."""
    plt.figure(figsize=(12, 4))
    plt.plot(time, E_J)
    if idx_max is not None:
        plt.scatter([time[idx_max]], [E_J[idx_max]], s=60, zorder=5)
        plt.axhline(float(E_J[idx_max]), linestyle="--")
        plt.text(
            0.56, 0.9, f"Buffer = {cap_J:,.1f} J",
            transform=plt.gca().transAxes,
        )
    plt.ylabel("Energy (J)")
    plt.title("Energy Trace")
    plt.tight_layout()
    plt.show()


# ── Self-test entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    inputs = Inputs(
        latitude=seattle_lat, longitude=seattle_long,
        active_area_m2=shore_pod_realistic_SA_m2, efficiency=sb_sca_efficiency,
        max_power_draw_W=0.25, start_date="2025-06-09", end_date="2025-06-22",
        location="Seattle", dt_minutes=1,
    )
    strategy = {
        "mode": "constant", "signal": "solar",
        "p_hibernate_W": 1e-9, "on_threshold_W": 0.30,
        "off_threshold_W": 0.10, "initial_state": "hibernate",
    }

    cap_J, cap_Wh, time, solar_W, load_W, net_W, E_J, idx_max, duty, signal_W, dni_W_m2 = \
        compute_required_buffer_max_accumulated_floor0(
            latitude=inputs.latitude, longitude=inputs.longitude,
            start_date=inputs.start_date, end_date=inputs.end_date,
            location=inputs.location, active_area_m2=inputs.active_area_m2,
            efficiency=inputs.efficiency, max_power_draw_W=inputs.max_power_draw_W,
            strategy=strategy, dt_minutes=inputs.dt_minutes, e0_J=2000, e_floor_J=1500,
        )

    print(f"Buffer capacity: {cap_J:.1f} J  ({cap_Wh:.3f} Wh)")
    if idx_max is not None:
        print(f"  Peak at: {time[idx_max]}")

    plot_traces(time, solar_W, load_W, net_W)
    plot_energy_trace_with_max(time, E_J, idx_max, cap_J)
