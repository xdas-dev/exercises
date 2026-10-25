"""Helpers given to the learners."""

import time

import numpy as np
import pandas as pd
import xdas as xd
from scipy.ndimage import gaussian_filter1d
from xdas.atoms import Atom, State

EARTH_RADIUS = 6371.0  # km


def distance_km(longitude1, latitude1, longitude2, latitude2):
    """Distance (km) along the Earth's surface between two points given in °.

    Works on numpy arrays too: one point against many, or many against many.
    """
    longitude1, latitude1, longitude2, latitude2 = map(np.deg2rad, [longitude1, latitude1, longitude2, latitude2])
    a = (
        np.sin((latitude2 - latitude1) / 2) ** 2
        + np.cos(latitude1) * np.cos(latitude2) * np.sin((longitude2 - longitude1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))


# The search grid of `locate`: about 5 km in every direction.
GRID = np.meshgrid(
    np.arange(-72.4, -71.2, 0.05),  # longitude (°)
    np.arange(-30.8, -29.0, 0.05),  # latitude (°)
    np.arange(0.0, 100.0, 5.0),  # depth (km)
    indexing="ij",
)
GRID = [g.ravel() for g in GRID]


def locate(picks, velocity=7.0, spread=0.2):
    """Locate an earthquake from P picks by grid search.

    For each grid point, every pick gives an origin time: pick time minus travel
    time at constant `velocity` (km/s). At the right location, all origin times
    agree. Their agreement is measured with a kernel density estimate (KDE) of
    width `spread` (s); the best grid point is the one with the highest peak.

    `picks` is a DataFrame with columns time, longitude, latitude, depth (m).
    Returns a Series with longitude, latitude, depth (m) and time (origin).
    """
    picks = picks.iloc[:: max(len(picks) // 400, 1)]  # ~400 picks are enough, and much faster
    reference = picks["time"].min()
    pt = (picks["time"] - reference).dt.total_seconds().values

    longitude, latitude, depth = GRID
    step = 0.05  # s
    bins = np.arange(-40.0, 10.0, step)  # origin times, relative to the first pick
    score = np.empty(longitude.size)
    origin = np.empty(longitude.size)
    for start in range(0, longitude.size, 2000):  # by blocks, to save memory
        block = slice(start, start + 2000)
        horizontal = distance_km(
            longitude[block, None], latitude[block, None], picks["longitude"].values, picks["latitude"].values
        )
        distance = np.sqrt(horizontal**2 + (depth[block, None] - picks["depth"].values / 1000) ** 2)
        origins = pt - distance / velocity  # one origin time per grid point and per pick
        # histogram of the origin times of each grid point, then smoothed: the KDE
        index = np.round((origins - bins[0]) / step).astype(int)
        inside = (index >= 0) & (index < bins.size)  # origins out of the bins do not count
        flat = (np.arange(len(origins))[:, None] * bins.size + index)[inside]
        density = np.bincount(flat, minlength=len(origins) * bins.size).reshape(len(origins), bins.size)
        density = gaussian_filter1d(density.astype(float), spread / step, axis=1)
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


class PeakAmplitude(Atom):
    """Pick with `detector`, and give each pick the peak amplitude that follows it.

    The input is the displacement, chunk by chunk. `detector` (STA/LTA then trigger)
    finds the picks. Each pick is held back until `window` seconds of data after it
    have arrived, then gets an `amplitude` column: the peak of |displacement| on its
    channel, between the pick and `window` seconds later. The last `keep` seconds of
    data are kept in memory for that (the trigger gives a pick a few seconds late).
    """

    def __init__(self, detector, window=2.0, keep=10.0):
        super().__init__()
        self.detector = detector
        self.window = window
        self.keep = keep
        self.buffer = State(...)  # the last `keep` seconds of |displacement|
        self.pending = State(...)  # the picks waiting for their window to end

    def initialize(self, da, **flags):
        self.buffer = State(None)
        self.pending = State(pd.DataFrame())

    def call(self, da, **flags):
        picks = self.detector(da, **flags)
        if picks is not None and len(picks):
            self.pending = State(pd.concat([self.pending, picks], ignore_index=True))
        buffer = np.abs(da) if self.buffer is None else xd.concatenate([self.buffer, np.abs(da)], dim="time")
        self.buffer = State(buffer)
        end = buffer["time"].values[-1]
        picks = self.release(end)
        self.buffer = State(buffer.sel(time=slice(end - np.timedelta64(int(self.keep * 1e9), "ns"), None)))
        return picks

    def release(self, end):
        """Measure and return the pending picks whose window is over at `end`."""
        if not len(self.pending):
            return pd.DataFrame()
        window = np.timedelta64(int(self.window * 1e9), "ns")
        ready = (self.pending["time"] + window <= end).values
        picks, self.pending = self.pending[ready].copy(), State(self.pending[~ready])
        times = self.buffer["time"].values
        channels = pd.Index(self.buffer["distance"].values).get_indexer(picks["distance"])
        start = np.searchsorted(times, picks["time"].values)
        stop = np.searchsorted(times, picks["time"].values + window, side="right")
        values = self.buffer.values
        picks["amplitude"] = [values[a:b, c].max() if b > a else np.nan for a, b, c in zip(start, stop, channels)]
        return picks

    def flush(self):
        if not self.initialized or self.buffer is None:
            return []
        for picks in self.detector.flush():
            self.pending = State(pd.concat([self.pending, picks], ignore_index=True))
        return [self.release(np.datetime64("2262-01-01"))]  # the end of the record: release everything

    def merge(self, results):
        """Stack the tables of the arrays of a collection (like `xd.trigger`)."""
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


class EEWS:
    """Earthquake early warning: locate (and size) earthquakes from a stream of picks.

    Call it with the new picks (a DataFrame with a `cable` column, maybe empty) and
    the current stream time `now`. Once a channel has picked, its next picks are
    ignored for `dead` seconds: they are the S wave or the coda, not a new P wave.
    Every `step` seconds, if at least `min_picks` channels have picked in the last
    `window` seconds, it locates the earthquake with these picks; otherwise the
    solution is NaN. If the picks have an `amplitude` column (peak P displacement, m),
    it also gives the magnitude: the median over the picks of ML, with the S amplitude
    taken as `scale` times the P amplitude.

    Returns a DataFrame with one row per solution computed during this call (maybe
    none): time (when it was computed), origin, longitude, latitude, depth (m),
    magnitude and picks (number of channels used). Stacked call after call, these
    rows show how the solution evolves with time.

    The picks need the longitude, latitude and depth of their channel. If they only
    have a distance, give the cables (a collection like `abyss`) as `geometry`: the
    positions are looked up there, at the nearest channel.
    """

    def __init__(self, window=10.0, step=1.0, min_picks=1000, dead=20.0, scale=7.0, geometry=None):
        self.window = pd.Timedelta(seconds=window)
        self.step = pd.Timedelta(seconds=step)
        self.min_picks = min_picks
        self.dead = pd.Timedelta(seconds=dead)
        self.scale = scale
        self.geometry = {}  # per cable: the distances, and the position of each channel
        for cable, da in (geometry or {}).items():
            positions = {name: da[name].values for name in ["longitude", "latitude", "depth"]}
            self.geometry[cable] = (pd.Index(da["distance"].values), positions)
        self.picks = pd.DataFrame()  # the picks kept, at most one per channel every `dead` s
        self.next = None  # when the next solution is due

    def __call__(self, picks, now):
        now = pd.Timestamp(now)
        if len(picks):
            self.add(picks)
        if len(self.picks):
            self.picks = self.picks[self.picks["time"] > now - max(self.window, self.dead)]
        if self.next is None:
            self.next = now.floor(self.step)
        if now < self.next:
            return pd.DataFrame()
        self.next = now.floor(self.step) + self.step
        return pd.DataFrame([self.solve(now)])

    def add(self, picks):
        """Keep the first of the new picks of each channel, unless it is in its dead time."""
        channel = ["cable", "distance"]
        picks = picks.sort_values("time").groupby(channel, as_index=False).first()
        if self.geometry:
            picks = self.position(picks)
        if len(self.picks):
            last = self.picks.groupby(channel)["time"].max()
            last = last.reindex(pd.MultiIndex.from_frame(picks[channel])).values
            picks = picks[~(picks["time"].values - last < self.dead)]  # never picked: NaT, kept
        self.picks = pd.concat([self.picks, picks], ignore_index=True)

    def position(self, picks):
        """Give each pick the longitude, latitude and depth of its channel, from `geometry`."""
        picks = picks.copy()
        for cable, rows in picks.groupby("cable"):
            distances, positions = self.geometry[cable]
            index = distances.get_indexer(rows["distance"], method="nearest")
            for name, values in positions.items():
                picks.loc[rows.index, name] = values[index]
        return picks

    def solve(self, now):
        solution = {"time": now, "origin": pd.NaT, "longitude": np.nan, "latitude": np.nan, "depth": np.nan}
        solution |= {"magnitude": np.nan, "picks": 0}
        if not len(self.picks):
            return solution
        picks = self.picks[self.picks["time"] > now - self.window]
        solution["picks"] = len(picks)
        if len(picks) < self.min_picks:
            return solution
        location = locate(picks)
        solution |= {"origin": location["time"]} | location.drop("time").to_dict()
        if "amplitude" in picks:
            d = distance_km(location["longitude"], location["latitude"], picks["longitude"], picks["latitude"])
            R = np.sqrt(d**2 + ((picks["depth"] - location["depth"]) / 1000) ** 2)
            A = self.scale * picks["amplitude"] * 1e9  # S amplitude (nm), from the P one
            solution["magnitude"] = np.nanmedian(np.log10(A) + 1.11 * np.log10(R) + 0.00189 * R - 2.09)
        return solution


def poll(subscribers):
    """Listen to several streams at once: yield (name, packet) as the packets arrive.

    `subscribers` is a dict of `ZMQSubscriber`. Never ends: break out of the loop.
    """
    for subscriber in subscribers.values():
        subscriber.timeout = 0  # next() returns at once, or raises TimeoutError
    while True:
        idle = True
        for name, subscriber in subscribers.items():
            while True:  # all the packets waiting, so that no stream falls behind the others
                try:
                    packet = next(subscriber)
                except TimeoutError:
                    break
                idle = False
                yield name, packet
        if idle:
            time.sleep(0.01)  # nothing waiting: do not spin


class Dashboard:
    """A live view of an early-warning run, for notebook 4.

    Left, the picks of the last `span` seconds on each cable. Right, the last solution on
    a map (red star) against the `reference` one (gold star), and below, how the solution
    evolves over the same last `span` seconds: its distance to the reference, and its
    magnitude. `reference` is a dict with
    longitude, latitude, depth (m) and magnitude.

    Show `dashboard` once (the last line of a cell), then call `update` with the new picks
    and solutions: the figure redraws in place, at most every `every` seconds. Solutions
    more than `gap` seconds apart belong to different events: their lines are not joined.
    Made for two cables.

    Drawing is the only work that can wait: when the packets lag more than `late` seconds
    behind the clock, the figure redraws only every `10 * every` seconds, so that the
    picking catches up. The title of the bottom right panel shows the lag (meaningless
    with `replay.py --speedup`, as the stream then runs ahead of the clock).
    """

    def __init__(self, geometry, reference, topo=None, span=60.0, every=0.5, gap=5.0, late=1.0):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        self.reference = reference
        self.span = pd.Timedelta(seconds=span)
        self.every = every
        self.gap = pd.Timedelta(seconds=gap)
        self.late = late
        self.cables = list(geometry)
        self.picks = pd.DataFrame(columns=["cable", "time", "distance"])
        self.solutions = pd.DataFrame()
        self.last = 0.0  # when the figure was last redrawn (wall clock)
        self.lag = 0.0  # the worst lag behind the clock since then (s)

        fig = make_subplots(  # two cables on the left, one above the other
            rows=2,
            cols=2,
            column_widths=[0.6, 0.4],
            specs=[[{}, {}], [{}, {"secondary_y": True}]],
            subplot_titles=[self.cables[0], "waiting for a solution", self.cables[1], "solution against reference"],
            horizontal_spacing=0.1,
        )
        for row, cable in enumerate(self.cables, start=1):
            fig.add_trace(go.Scattergl(name=f"picks {cable}", mode="markers", marker=dict(size=2, color="black")), row, 1)
            fig.update_yaxes(title_text="distance (km)", row=row, col=1)

        # the map
        if topo is not None:
            for line in _coastline(topo):
                fig.add_trace(go.Scatter(x=line[:, 0], y=line[:, 1], mode="lines", line=dict(color="gray", width=1)), 1, 2)
        for cable, da in geometry.items():
            da = da.isel(distance=slice(None, None, 10))  # enough for the map, and light to redraw
            fig.add_trace(go.Scatter(x=da["longitude"].values, y=da["latitude"].values, mode="lines", line=dict(color="red", width=3)), 1, 2)
        star = dict(symbol="star", size=18, line=dict(color="black", width=1))
        fig.add_trace(go.Scatter(x=[reference["longitude"]], y=[reference["latitude"]], mode="markers", marker=star | dict(color="gold")), 1, 2)
        fig.add_trace(go.Scatter(name="solution", x=[], y=[], mode="markers", marker=star | dict(color="red")), 1, 2)
        fig.update_xaxes(range=[-72.3, -71.1], row=1, col=2)
        fig.update_yaxes(range=[-31.0, -29.1], scaleanchor="x2", scaleratio=1 / np.cos(np.deg2rad(30)), row=1, col=2)

        # the evolution of the solution
        fig.add_trace(go.Scatter(name="error (km)", mode="lines+markers", line=dict(color="black")), 2, 2)
        fig.add_trace(go.Scatter(name="magnitude", mode="lines+markers", line=dict(color="red")), 2, 2, secondary_y=True)
        fig.add_trace(go.Scatter(name="reference magnitude", mode="lines", line=dict(color="red", dash="dash")), 2, 2, secondary_y=True)
        fig.update_yaxes(title_text="distance to reference (km)", rangemode="tozero", row=2, col=2)
        fig.update_yaxes(title_text="ML", range=[2, 5], secondary_y=True, row=2, col=2)

        fig.update_layout(height=650, showlegend=False, margin=dict(l=60, r=60, t=40, b=40))
        self.figure = fig  # a plain figure, not a widget: works in JupyterLab and VS Code alike
        self.trace = {trace.name: trace for trace in self.figure.data if trace.name}
        self.handle = None  # where the figure is shown, to redraw it in place

    def _ipython_display_(self):
        from IPython.display import display

        # the raw plotly output: a figure shown as itself ignores display_id, so could not be redrawn
        self.handle = display(self.figure._repr_mimebundle_(), raw=True, display_id=True)

    def update(self, picks, solutions, now):
        """Add the new picks (with a `cable` column) and solutions (from `EEWS`), and redraw."""
        now = pd.Timestamp(now)
        if len(picks):
            self.picks = pd.concat([self.picks, picks[["cable", "time", "distance"]]], ignore_index=True)
        if len(solutions):
            self.solutions = pd.concat([self.solutions, solutions], ignore_index=True)
        self.lag = max(self.lag, (pd.Timestamp(time.time_ns()) - now).total_seconds())  # stamps are UTC
        every = self.every if self.lag < self.late else 10 * self.every  # behind: let the picking catch up
        if time.monotonic() - self.last < every:
            return
        self.figure.layout.annotations[3].text = (
            f"solution against reference (lag {self.lag:.1f} s{', display slowed down' if self.lag >= self.late else ''})"
        )
        self.last, self.lag = time.monotonic(), 0.0
        self.picks = self.picks[self.picks["time"] > now - self.span]

        for row, cable in enumerate(self.cables, start=1):
            mine = self.picks[self.picks["cable"] == cable]
            self.trace[f"picks {cable}"].update(x=mine["time"], y=mine["distance"] / 1000)
            self.figure.update_xaxes(range=[now - self.span, now], row=row, col=1)
        self.figure.update_xaxes(range=[now - self.span, now], row=2, col=2)  # the solution, on the same clock
        found = self.solutions.dropna(subset=["longitude"]) if len(self.solutions) else self.solutions
        if len(found):
            last = found.iloc[-1]
            self.trace["solution"].update(x=[last["longitude"]], y=[last["latitude"]])
            d = distance_km(self.reference["longitude"], self.reference["latitude"], found["longitude"], found["latitude"])
            curves = pd.DataFrame({
                "time": found["time"],
                "error": np.sqrt(d**2 + ((found["depth"] - self.reference["depth"]) / 1000) ** 2),
                "magnitude": found["magnitude"],
                "reference": self.reference["magnitude"],
            })
            # a NaN point between two events: plotly does not join the lines across it
            start = curves["time"].diff() > self.gap
            breaks = pd.DataFrame({"time": curves["time"][start] - pd.Timedelta(1, "ms")})
            curves = pd.concat([curves, breaks]).sort_values("time")
            self.trace["error (km)"].update(x=curves["time"], y=curves["error"])
            self.trace["magnitude"].update(x=curves["time"], y=curves["magnitude"])
            self.trace["reference magnitude"].update(x=curves["time"], y=curves["reference"])
            self.figure.layout.annotations[1].text = (
                f"{last['depth'] / 1000:.0f} km deep, origin {str(last['origin'])[11:22]}, ML {last['magnitude']:.1f}"
            )
        if self.handle is not None:
            self.handle.update(self.figure._repr_mimebundle_(), raw=True)  # redraw in place


def _coastline(topo):
    """The coastline (0 m contour) of a topography, as a list of (n, 2) arrays of longitude, latitude."""
    from contourpy import contour_generator

    topo = topo[::4, ::4].load()
    generator = contour_generator(topo["longitude"].values, topo["latitude"].values, topo.values.astype(float))
    return [line for line in generator.lines(0.0) if len(line) > 20]  # skip the islets
