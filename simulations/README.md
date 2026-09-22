# Shore Pod — Solar Energy Harvesting Simulation

Simulation code for the Shore Pod oceanographic sensor platform. Given historical irradiance data for a target deployment location, the simulation sizes an onboard energy buffer and evaluates operating strategies to maximize uptime. Irradiance data is fetched automatically from the [Open-Meteo](https://open-meteo.com/) historical archive.

Accompanies the paper:
> Amber Skotak and Bradley Denby. "Shore Pods: A Small, Low-Cost Platform for Energy Harvesting and Maritime Sensing." *IEEE OCEANS*, 2026.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Reproducing figures

**Energy-scheduling pitfalls** (Figure 7):
```bash
python failure_modes.py
```
Displays three subplots quantifying uptime lost to buffer overflow, buffer depletion, and unnecessary hibernation.

**Buffer sizing, threshold sweep, strategy traces, and uptime comparison** (Figures 6, 8, 9, 10):
```bash
python Threshold_study.py
```
Displays four figures in sequence:
1. **Buffer sizing** — solar power trace and accumulated net energy used to determine minimum buffer capacity
2. **Uptime heatmap** — uptime across 5 050 hibernation entrance/exit threshold combinations
3. **Stacked strategy traces** — per-strategy buffer energy over the two-week Intermittent Energy case
4. **Uptime % of maximum** — clustered bar chart comparing all four strategies across five deployment environments

---

## File overview

| File | Description |
|------|-------------|
| `config.py` | Hardware constants, site coordinates, operating strategies, and environmental test cases |
| `irradiance.py` | Open-Meteo API wrappers — hourly and minute-resolution DNI fetching with local caching |
| `buffer_sizing.py` | Buffer capacity sizing pipeline (`compute_required_buffer_max_accumulated_floor0`) |
| `Threshold_study.py` | Strategy simulation and all four analysis figures |
| `failure_modes.py` | Three-panel failure mode visualization |

---

## Data

Irradiance is fetched from the Open-Meteo archive API on first run and cached in `.cache/` for reuse. No data files need to be downloaded manually.

---

## Hardware

The Shore Pod uses SCA solar cells with:
- Active area: `0.00245 m²`
- Panel efficiency: 25%
- Active power draw: 0.25 W
- Buffer capacity: sized per deployment location by `buffer_sizing.py`

---

## Deployment environments

| Name | Location | Period |
|------|----------|--------|
| Intermittent Energy | Seattle, WA | Jun 9–22, 2025 |
| Energy Scarce | Seattle, WA | Dec 1–14, 2025 |
| Uniform Energy | Panama Canal, Panama | Mar 1–14, 2025 |
| Cloud-Driven Energy | Paradise Bay, Antarctica | Dec 1–14, 2025 |
| Energy Abundant | Port Hedland, Australia | Jan 22–Feb 4, 2025 |
