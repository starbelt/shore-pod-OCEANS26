"""
config.py — Shore Pod Solar Simulation: Hardware constants, site coordinates,
operating strategies, and environmental test cases.

All simulation parameters that are not algorithm logic live here so that
reproduction of paper figures requires changing only this file.
"""

# ── Geographic coordinates ─────────────────────────────────────────────────────
seattle_lat                  =  47.6062
seattle_long                 = -122.3321
paradise_bay_antarctica_lat  = -64.8574
paradise_bay_antarctica_long = -62.9201
port_hedland_aus_lat         = -20.3111
port_hedland_aus_long        = 118.6094
panama_canal_lat             =   9.38743
panama_canal_long            = -79.91863

# ── Shore Pod hardware constants ───────────────────────────────────────────────
sb_sca_active_area        = 0.000978048   # m², active cell area per SCA board
sb_sca_efficiency         = 0.25          # solar cell efficiency (25 %)
shore_pod_realistic_SA_m2 = sb_sca_active_area / 8 * 20  # scaled to Shore Pod body area

# ── Operating strategies ───────────────────────────────────────────────────────
# Each strategy drives the hibernation simulation in threshold_study.py.
#   "energy_blind"  — fixed duty cycle, no buffer awareness
#   "hib"           — Schmitt-trigger hibernation (enter/exit thresholds as
#                     fractions of buffer capacity)
STRATEGIES = {
    "Energy blind (50% Duty Cycle)": {
        "type": "energy_blind",
        "duty": 0.5,
        "sleeps_per_hour": 1 / 12,
        "E_low_frac": 0.05,
        "E_high_frac": 0.6,
        "start_phase_hours": 0.0,
    },
    "Lazy": {
        "type": "hib",
        "enter_frac": 0.5,
        "exit_frac": 0.8,
    },
    "Frugal": {
        "type": "hib",
        "enter_frac": 0.1,
        "exit_frac": 0.8,
    },
    "Generous": {
        "type": "hib",
        "enter_frac": 0.1,
        "exit_frac": 0.15,
    },
}

# ── Environmental test cases ───────────────────────────────────────────────────
# Five representative deployments spanning the global energy-availability spectrum.
GRID_CASES = [
    {
        "name":  "Cloud-Driven Energy",
        "lat":   paradise_bay_antarctica_lat,
        "lon":   paradise_bay_antarctica_long,
        "start": "2025-12-01",
        "end":   "2025-12-14",
    },
    {
        "name":  "Intermittent Energy",
        "lat":   seattle_lat,
        "lon":   seattle_long,
        "start": "2025-06-09",
        "end":   "2025-06-22",
    },
    {
        "name":  "Energy Scarce",
        "lat":   seattle_lat,
        "lon":   seattle_long,
        "start": "2025-12-01",
        "end":   "2025-12-14",
    },
    {
        "name":  "Energy Abundant",
        "lat":   port_hedland_aus_lat,
        "lon":   port_hedland_aus_long,
        "start": "2025-01-22",
        "end":   "2025-02-04",
    },
    {
        "name":  "Uniform Energy",
        "lat":   panama_canal_lat,
        "lon":   panama_canal_long,
        "start": "2025-03-01",
        "end":   "2025-03-14",
    },
]
