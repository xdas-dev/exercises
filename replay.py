"""Replay SER/N (tcp://*:5556) and SER/S (tcp://*:5557) in a loop.

    uv run replay.py --speedup 4

Each cable is streamed as a live ASN OptoDAS interrogator would do it: packets
of 24 samples, sent once recorded and stamped with the current UTC time. Read
them with `xdas.io.asn.ZMQSubscriber`.

The 240 s records are played in an endless 210 s loop: the last 30 s are faded
into the first ones to hide the seam. Times stay continuous across loops, and
both cables keep their relative timing, so the earthquake comes back every
210 s with its real moveout. `--speedup` plays that many times faster than
real time.
"""

import argparse
from itertools import cycle
from time import sleep, time_ns

import numpy as np
import xdas as xd
from xdas.io.asn import ZMQPublisher

TREF = np.datetime64("2024-09-02T02:55:00")
FADE = np.timedelta64(30, "s")  # crossfade
DURATION = np.timedelta64(210, "s")  # loop length: the 240 s record minus the fade
PACKET_SIZE = 24  # samples: 0.24 s on N, 0.384 s on S


def make_packets(da):
    """Turn a record into a seamless loop, cut into packets timed from TREF."""
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
    da = xd.DataArray(data, {"time": t, "distance": da["distance"]})

    return xd.split(da, len(data) // PACKET_SIZE)


def stream(packets, t0):
    """The packets in an endless loop, with continuous times from `t0`."""
    for count, packet in enumerate(cycle(packets)):
        offset = t0 + (count // len(packets)) * DURATION
        t = packet["time"] + offset
        yield xd.DataArray(packet.values, {"time": t, "distance": packet["distance"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--speedup",
        type=int,
        default=1,
        help="times faster than real time (default: 1)",
    )
    speedup = parser.parse_args().speedup

    das = xd.open("data/das/{cable}/[record].hdf5", squeeze=True)
    packets = {cable: make_packets(das[cable]) for cable in ["N", "S"]}
    ports = {"N": 5556, "S": 5557}
    publishers = {
        cable: ZMQPublisher(f"tcp://*:{port}") for cable, port in ports.items()
    }
    for cable, port in ports.items():
        print(f"SER/{cable} -> tcp://localhost:{port}")
    print(f"Looping every {DURATION}, {speedup}x real time. Ctrl-C to stop.")

    t0 = np.datetime64("now", "s")  # stream time starts now (UTC)
    streams = {cable: stream(packets[cable], t0) for cable in packets}
    pending = {cable: next(streams[cable]) for cable in streams}  # next packet of each
    try:
        while True:
            now = np.datetime64(time_ns(), "ns")
            for cable, packet in pending.items():
                # released once recorded: when the stream time reaches its end
                delta = xd.get_sampling_interval(packet, "time", cast=False)
                end = packet["time"][-1].values + delta
                if speedup * (now - t0) >= end - t0:
                    publishers[cable].submit(packet)
                    pending[cable] = next(streams[cable])
            sleep(0.001)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        for publisher in publishers.values():
            publisher.close()


if __name__ == "__main__":
    main()
