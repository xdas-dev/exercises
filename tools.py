"""Helpers given to the learners."""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

KM_PER_DEGREE = 111.19
REFERENCE_LATITUDE = -30.0  # flat-earth projection, good enough at this scale


def to_km(longitude, latitude):
    """Project longitude, latitude (°) to x, y (km) on a flat earth."""
    x = longitude * KM_PER_DEGREE * np.cos(np.deg2rad(REFERENCE_LATITUDE))
    y = latitude * KM_PER_DEGREE
    return x, y


def locate(picks, velocity=7.0, spread=0.2):
    """Locate an earthquake from P picks by grid search.

    For each grid point, every pick gives an origin time: pick time minus travel
    time at constant `velocity` (km/s). At the right location, all origin times
    agree. Their agreement is measured with a kernel density estimate (KDE) of
    width `spread` (s); the best grid point is the one with the highest peak.

    `picks` is a DataFrame with columns time, longitude, latitude, depth (m).
    Returns a Series with longitude, latitude, depth (m) and time (origin).
    """
    picks = picks.iloc[::4]  # one pick out of four is enough, and four times faster
    px, py = to_km(picks["longitude"].values, picks["latitude"].values)
    pz = picks["depth"].values / 1000
    reference = picks["time"].min()
    pt = (picks["time"] - reference).dt.total_seconds().values

    longitudes = np.arange(-72.4, -71.2, 0.02)
    latitudes = np.arange(-30.8, -29.0, 0.02)
    depths = np.arange(0.0, 100.0, 5.0)
    grid = np.meshgrid(longitudes, latitudes, depths, indexing="ij")
    longitude, latitude, depth = [g.ravel() for g in grid]
    x, y = to_km(longitude, latitude)

    step = 0.05  # s
    bins = np.arange(-40.0, 10.0, step)  # origin times, relative to the first pick
    score = np.empty(longitude.size)
    origin = np.empty(longitude.size)
    for start in range(0, longitude.size, 5000):  # by blocks, to save memory
        block = slice(start, start + 5000)
        distance = np.sqrt((x[block, None] - px) ** 2 + (y[block, None] - py) ** 2 + (depth[block, None] - pz) ** 2)
        origins = pt - distance / velocity
        index = np.clip(np.round((origins - bins[0]) / step).astype(int), 0, bins.size - 1)
        density = np.zeros((index.shape[0], bins.size))
        np.add.at(density, (np.arange(index.shape[0])[:, None], index), 1.0)
        density = gaussian_filter1d(density, spread / step, axis=1)
        score[block] = density.max(axis=1)
        origin[block] = bins[density.argmax(axis=1)]

    best = score.argmax()
    return pd.Series(
        {
            "longitude": longitude[best],
            "latitude": latitude[best],
            "depth": depth[best] * 1000,
            "time": reference + pd.Timedelta(seconds=origin[best]),
        }
    )


class Locator:
    """Locate earthquakes from a stream of picks, one chunk of picks at a time.

    Like an atom, it keeps a memory between calls: the picks not used yet. Call it
    with the new picks (a DataFrame with a `cable` column, maybe empty) and the
    current stream time `now`. An earthquake starts with a burst: at least
    `burst` picks within 2 s. Earlier picks are dropped as noise. Once `wait`
    seconds have passed since the burst, the first pick of each channel in that
    time window is used to locate the earthquake. Returns the location, or None.
    """

    def __init__(self, wait=20.0, burst=50):
        self.wait = pd.Timedelta(seconds=wait)
        self.burst = burst
        self.pending = pd.DataFrame()

    def __call__(self, picks, now):
        self.pending = pd.concat([self.pending, picks], ignore_index=True).sort_values("time")
        while len(self.pending):
            start = self.pending["time"].iloc[0]
            if pd.Timestamp(now) - start < pd.Timedelta(seconds=2):
                return None  # too early to tell
            if (self.pending["time"] < start + pd.Timedelta(seconds=2)).sum() < self.burst:
                self.pending = self.pending.iloc[1:]  # a lonely pick: noise
                continue
            if pd.Timestamp(now) - start < self.wait:
                return None  # wait for the waves to cross the cables
            selected = self.pending["time"] < start + self.wait
            window = self.pending[selected].groupby(["cable", "distance"], as_index=False).first()
            self.pending = self.pending[~selected]
            location = locate(window)
            location["picks"] = len(window)
            return location
        return None
