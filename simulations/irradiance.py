"""
Irradiance data fetching from the Open-Meteo historical archive API.

Functions
---------
get_irradiance_data          — hourly DNI + cloud cover for a location/date range
get_minute_irradiance_scaled — minute-resolution irradiance with optional wave-rocking model
get_minute_irradiance        — minute-resolution DNI via linear interpolation
"""

import openmeteo_requests
import requests_cache
from retry_requests import retry

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _make_client():
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def get_irradiance_data(latitude, longitude, start_date, end_date, location):
    """
    Fetch hourly cloud cover and direct normal irradiance (DNI) from Open-Meteo.

    Returns
    -------
    hourly_dataframe : pd.DataFrame  — columns: date, cloud_cover, direct_normal_irradiance
    time             : float         — total hours in the requested period
    hourly_direct_normal_irradiance : np.ndarray
    dark_hours       : int           — hours where DNI ≤ 0.1 W/m²
    """
    openmeteo = _make_client()
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["cloud_cover", "direct_normal_irradiance"],
        "timezone": "auto",
    }
    responses = openmeteo.weather_api(url, params=params)

    start_dt = datetime.strptime(params["start_date"], "%Y-%m-%d")
    end_dt   = datetime.strptime(params["end_date"],   "%Y-%m-%d")
    time     = timedelta.total_seconds(end_dt - start_dt) / 3600 + 24

    response = responses[0]
    hourly   = response.Hourly()
    hourly_cloud_cover              = hourly.Variables(0).ValuesAsNumpy()
    hourly_direct_normal_irradiance = hourly.Variables(1).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time()    + response.UtcOffsetSeconds(), unit="s", utc=True),
            end=  pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        "cloud_cover":              hourly_cloud_cover,
        "direct_normal_irradiance": hourly_direct_normal_irradiance,
    }
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    dark_hours = int((hourly_dataframe["direct_normal_irradiance"] <= 0.1).sum())
    return hourly_dataframe, time, hourly_direct_normal_irradiance, dark_hours


def get_minute_irradiance_scaled(
    latitude, longitude, start_date, end_date, location,
    dt_minutes=1, dni_threshold=5.0, mode="scaled", rock_amplitude_deg=30.0,
):
    """
    Fetch hourly DNI and direct_radiation from Open-Meteo and return irradiance
    appropriate for a vertical panel on a floating platform.

    mode="scaled"     — DNI × cos(zenith) = direct_radiation  [upright, no rocking]
    mode="mean_wave"  — cycle-averaged rocking: DNI × <max(0, cos(z − A·sin(ωt)))>
    mode="worst_wave" — worst-case tilt: DNI × max(0, cos(z + rock_amplitude))

    Returns (time_min, irr_min) at dt_minutes resolution.
    """
    openmeteo = _make_client()
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["direct_normal_irradiance", "direct_radiation"],
        "timezone": "auto",
    }
    response = openmeteo.weather_api(url, params=params)[0]
    hourly   = response.Hourly()

    t_hourly = pd.date_range(
        start=pd.to_datetime(hourly.Time()    + response.UtcOffsetSeconds(), unit="s", utc=True),
        end=  pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_localize(None)

    dni        = hourly.Variables(0).ValuesAsNumpy().astype(float)
    direct_rad = hourly.Variables(1).ValuesAsNumpy().astype(float)

    sun_up = dni > 1.0
    cos_z  = np.where(sun_up, np.clip(direct_rad / np.where(sun_up, dni, 1.0), 0.0, 1.0), 0.0)

    if mode == "scaled":
        irr_hourly = np.where(sun_up, np.maximum(0.0, direct_rad), 0.0)
    elif mode == "mean_wave":
        A       = np.radians(float(rock_amplitude_deg))
        zenith  = np.arccos(cos_z)
        t_cyc   = np.linspace(0, 2 * np.pi, 10_000, endpoint=False)
        rock    = A * np.sin(t_cyc)
        incidence   = zenith[:, None] - rock[None, :]
        mean_factor = np.maximum(0.0, np.cos(incidence)).mean(axis=1)
        irr_hourly  = np.where(sun_up, dni * mean_factor, 0.0)
    elif mode == "worst_wave":
        A          = np.radians(float(rock_amplitude_deg))
        zenith     = np.arccos(cos_z)
        min_factor = np.maximum(0.0, np.cos(zenith + A))
        irr_hourly = np.where(sun_up, dni * min_factor, 0.0)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'scaled', 'mean_wave', or 'worst_wave'.")

    t_min = pd.date_range(start=t_hourly[0], end=t_hourly[-1], freq=f"{dt_minutes}min")
    irr_min = np.interp(t_min.view("int64"), t_hourly.view("int64"), irr_hourly)
    irr_min[irr_min < float(dni_threshold)] = 0.0
    return t_min, irr_min


def get_minute_irradiance(
    latitude, longitude, start_date, end_date,
    location, dt_minutes=1, dni_threshold=5.0,
):
    """
    Fetch hourly DNI and linearly interpolate to minute resolution.

    Returns (t_min, dni_min).
    """
    hourly_df, _, hourly_dni, _ = get_irradiance_data(
        latitude, longitude, start_date, end_date, location
    )
    t_hourly  = pd.to_datetime(hourly_df["date"], utc=True).dt.tz_localize(None)
    dni_hourly = np.asarray(hourly_dni, dtype=float)

    t_min = pd.date_range(start=t_hourly.iloc[0], end=t_hourly.iloc[-1], freq=f"{dt_minutes}min")
    dni_min = np.interp(t_min.view("int64"), t_hourly.view("int64"), dni_hourly)
    dni_min[dni_min < dni_threshold] = 0.0
    return t_min, dni_min
