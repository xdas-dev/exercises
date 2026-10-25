"""Replay SER/N (tcp://*:5556) and SER/S (tcp://*:5557) in a loop.

    uv run replay.py --speedup 4
    uv run replay.py --waterfall

Each cable is streamed as a live ASN OptoDAS interrogator would do it: packets
of 24 samples, sent once recorded and stamped with the current UTC time. Read
them with `xdas.io.asn.ZMQSubscriber`.

The 240 s records are played in an endless 210 s loop: the last 30 s are faded
into the first ones to hide the seam. Times stay continuous across loops, and
both cables keep their relative timing, so the earthquake comes back every
210 s with its real moveout. `--speedup` plays that many times faster than
real time. `--waterfall` also shows the last minute of both cables in a window.
Do not combine them: the window is redrawn at every packet, which cannot keep up
with a faster stream, so the packets lag and come in bursts.

Press `e` (in the terminal or in the waterfall window) to jump to just before the
earthquake. The stream times go on without a gap, so a pipeline keeps its state:
only the data jump, a few seconds before the first arrival.
"""

import argparse
import select
import sys
import termios
import tty
from time import sleep, time_ns

import numpy as np
import scipy.signal as sp
import xdas as xd
from xdas.io.asn import ZMQPublisher

TREF = np.datetime64("2024-09-02T02:55:00")
FADE = np.timedelta64(30, "s")  # crossfade
DURATION = np.timedelta64(210, "s")  # loop length: the 240 s record minus the fade
PACKET_SIZE = 24  # samples: 0.24 s on N, 0.384 s on S
JUMP = np.datetime64("2024-09-02T02:56:00")  # where `e` jumps to: ~13 s before the first arrival
JUMP_FADE = np.timedelta64(2, "s")  # crossfade of a jump: a cut would trigger picks


def make_loop(da):
    """Turn a record into a seamless loop, timed from TREF."""
    da = da.load()
    nfade = FADE // xd.get_sampling_interval(da, "time", cast=False)
    data = da.values
    cross = (
        np.linspace(0, 1, nfade)[:, None] * data[:nfade]
        + np.linspace(1, 0, nfade)[:, None] * data[-nfade:]
    )
    # half of the crossfade at the start of the loop, half at its end
    data = np.concatenate(
        [cross[nfade // 2 :], data[nfade:-nfade], cross[: nfade // 2]]
    )

    start = nfade // 2
    t = da["time"][start : start + len(data)] - TREF
    return xd.DataArray(data, {"time": t, "distance": da["distance"]})


class Loop:
    """A loop played as an endless stream of packets, with continuous times from `t0`."""

    def __init__(self, da, t0):
        self.data = da.values
        self.distance = da["distance"]
        self.delta = xd.get_sampling_interval(da, "time", cast=False)
        self.first = da["time"][0].values  # loop time of the first sample (from TREF)
        self.position = 0  # in the loop, of the next sample
        self.time = t0 + self.first  # stream time of the next sample
        self.fade = None  # after a jump: (where the loop was, samples faded so far)
        self.next = None  # the next packet, once made

    def where(self):
        """The loop time (from TREF) of the next sample."""
        return self.first + self.position * self.delta

    def peek(self):
        """The next packet, with its stream times."""
        if self.next is None:
            index = np.arange(PACKET_SIZE)
            data = self.data[(self.position + index) % len(self.data)]
            if self.fade is not None:  # from where the loop was to where it is now
                old, done = self.fade
                weight = np.clip((done + index) / (JUMP_FADE / self.delta), 0, 1)[:, None] * np.pi / 2
                before = self.data[(old + index) % len(self.data)]
                data = np.sin(weight) * data + np.cos(weight) * before  # keeps the noise level
            time = xd.InterpCoordinate.from_block(self.time, PACKET_SIZE, self.delta, dim="time")
            self.next = xd.DataArray(data, {"time": time, "distance": self.distance})
        return self.next

    def advance(self):
        self.position = (self.position + PACKET_SIZE) % len(self.data)
        self.time = self.time + PACKET_SIZE * self.delta
        if self.fade is not None:
            old, done = self.fade
            done += PACKET_SIZE
            self.fade = None if done >= JUMP_FADE / self.delta else ((old + PACKET_SIZE) % len(self.data), done)
        self.next = None

    def skip(self, shift):
        """Jump `shift` further in the loop, fading in over JUMP_FADE. Returns the shift done.

        The stream times go on as before: only the data jump.
        """
        samples = round(shift / self.delta)
        self.fade = (self.position, 0)
        self.position = (self.position + samples) % len(self.data)
        self.next = None
        return samples * self.delta


def keystroke():
    """The key pressed in the terminal, if any (without waiting), else None."""
    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


class Waterfall:
    """The last minute of both cables, in a window that updates itself.

    For the eye only: the channels and the times are taken one out of n, to draw an
    image of about PIXELS; each packet has its common mode removed and is band-passed
    (2-15 Hz). The colors show the amplitude, on a fixed log scale (CLIM). White: no
    data yet.
    """

    SPAN = 60.0  # s, height of the window
    PIXELS = (300, 200)  # (time, distance): about the size of each image
    CLIM = (0.01, 30.0)  # amplitude (raw units): the noise is ~0.01, the earthquake up to ~30

    def __init__(self, das):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        self.plt = plt
        self.fig, axes = plt.subplots(1, len(das), figsize=(12, 6))
        self.buffers, self.filters, self.images, self.steps = {}, {}, {}, {}
        self.counts = {cable: 0 for cable in das}  # samples received, to decimate at fixed times
        for ax, (cable, da) in zip(axes, das.items()):
            rate = 1 / xd.get_sampling_interval(da, "time")
            ntime = round(self.SPAN * rate)
            self.steps[cable] = (
                max(ntime // self.PIXELS[0], 1),
                max(da.sizes["distance"] // self.PIXELS[1], 1),
            )
            distance = da["distance"].values[:: self.steps[cable][1]] / 1000
            self.buffers[cable] = np.full((ntime, len(distance)), np.nan)  # no data yet
            sos = sp.butter(4, (2.0, 15.0), "bandpass", fs=rate, output="sos")
            self.filters[cable] = (sos, np.zeros((sos.shape[0], 2, len(distance))))
            self.images[cable] = ax.imshow(
                self.buffers[cable],
                aspect="auto",
                cmap=plt.get_cmap("viridis").with_extremes(bad="white"),  # no data
                norm=LogNorm(*self.CLIM),
                interpolation="none",
                extent=(distance[0], distance[-1], 0, self.SPAN),  # oldest at the top
            )
            ax.set_title(f"SER/{cable}")
            ax.set_xlabel("distance (km)")
        axes[0].set_ylabel("seconds ago")
        self.fig.tight_layout()
        self.key = None  # the last key pressed in the window
        self.fig.canvas.mpl_connect("key_press_event", lambda event: setattr(self, "key", event.key))
        plt.show(block=False)

    def add(self, cable, packet):
        data = packet.values[:, :: self.steps[cable][1]]
        data = data - np.median(data, axis=1, keepdims=True)  # common mode
        sos, zi = self.filters[cable]
        data, zi = sp.sosfilt(sos, data, axis=0, zi=zi)
        self.filters[cable] = (sos, zi)
        buffer = self.buffers[cable]
        self.buffers[cable] = np.concatenate([buffer[len(data) :], data])
        self.counts[cable] += len(data)

    def draw(self):
        for cable, image in self.images.items():
            # one sample out of `step`, always the same ones: no flicker from draw to draw
            buffer, step = self.buffers[cable], self.steps[cable][0]
            first = -(self.counts[cable] - len(buffer)) % step
            image.set_data(np.abs(buffer[first::step][-(len(buffer) // step) :]))
        # redraw without plt.pause, which brings the window to the front every time
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)

    def closed(self):
        return not self.plt.fignum_exists(self.fig.number)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--speedup",
        type=int,
        default=1,
        help="times faster than real time (default: 1)",
    )
    parser.add_argument(
        "--waterfall",
        action="store_true",
        help="show the last minute of both cables in a window (lags with --speedup)",
    )
    args = parser.parse_args()
    speedup = args.speedup

    das = xd.open("data/das/{cable}/[record].hdf5", squeeze=True)
    records = {cable: make_loop(das[cable]) for cable in ["N", "S"]}
    ports = {"N": 5556, "S": 5557}
    publishers = {
        cable: ZMQPublisher(f"tcp://*:{port}") for cable, port in ports.items()
    }
    for cable, port in ports.items():
        print(f"SER/{cable} -> tcp://localhost:{port}")
    print(f"Looping every {DURATION}, {speedup}x real time. Ctrl-C to stop.")
    print("Press e to jump to just before the earthquake.")
    waterfall = Waterfall({cable: das[cable] for cable in records}) if args.waterfall else None

    t0 = np.datetime64("now", "s")  # stream time starts now (UTC)
    loops = {cable: Loop(records[cable], t0) for cable in records}
    sent = {cable: 0 for cable in loops}
    terminal = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    if terminal:
        tty.setcbreak(sys.stdin)  # keys come without Enter
    try:
        while True:
            key = keystroke()
            if waterfall is not None and waterfall.key:
                key, waterfall.key = waterfall.key, None
            if key == "e":
                # the same shift on both cables, so they keep their relative timing
                shift = (JUMP - TREF) - loops["S"].where()
                for loop in loops.values():
                    loop.skip(shift)
                print("\nJumped to just before the earthquake.")

            now = np.datetime64(time_ns(), "ns")
            fresh = False  # whether a packet was sent in this turn
            for cable, loop in loops.items():
                packet = loop.peek()
                # released once recorded: when the stream time reaches its end
                delta = xd.get_sampling_interval(packet, "time", cast=False)
                end = packet["time"][-1].values + delta
                if speedup * (now - t0) >= end - t0:
                    publishers[cable].submit(packet)
                    loop.advance()
                    sent[cable] += 1
                    fresh = True
                    if waterfall is not None:
                        waterfall.add(cable, packet)
            if fresh:  # show it at once: the count on the same line, then the waterfall
                counts = ", ".join(f"SER/{cable} {count}" for cable, count in sent.items())
                print(f"\rPackets sent: {counts}", end="", flush=True)
                if waterfall is not None:
                    if waterfall.closed():
                        waterfall = None  # the window was closed: keep streaming
                    else:
                        waterfall.draw()
            sleep(0.001)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if terminal:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, terminal)  # the terminal as it was
        for publisher in publishers.values():
            publisher.close()


if __name__ == "__main__":
    main()
