# Xdas exercises: an offshore earthquake on two DAS cables

Hands-on exercises to learn the [Xdas](https://github.com/xdas-dev/xdas)
Python library, a library designed to facilitate the processing and analysis
of DAS data. Where the [tutorials](https://github.com/xdas-dev/tutorials)
show you how things are done, here you write the code yourself: each exercise
leaves a few lines (or just a value, `...`) for you to fill, gives a short
reminder of the syntax, and points to the page of the
[documentation](https://xdas.readthedocs.io/en/stable/) that helps.

The dataset is minimal: four minutes of DAS on the two SER submarine cables
off La Serena (Chile), recorded during the ABYSS project, and the ML 3.4
earthquake that occurred between them on 2024-09-02 at 02:56 UTC.

| Notebook | What you do |
|---|---|
| [1](1_preparation.ipynb) | Open the DAS records, inspect them, attach the cable geometry, draw a map |
| [2](2_exploration.ipynb) | Physical units, FK analysis, filtering, Wood–Anderson, STA/LTA picking, location and magnitude |
| [3](3_chunks.ipynb) | Turn the processing into atoms, write a continuous STA/LTA, pick chunk by chunk, locate from a stream of picks |
| [4](4_realtime.ipynb) | Real time: receive packets over ZeroMQ, pick them as they arrive, draw a live waterfall and locate |

Run them in order: notebook 1 writes `abyss.nc`, which the next ones read.
The course takes about three hours.

## Setup

Everything is done by one command. It needs [uv](https://docs.astral.sh/uv/),
which installs itself in one line too:

```
curl -LsSf https://astral.sh/uv/install.sh | sh    # macOS and Linux
```

Then clone the exercises and run the installer:

```
git clone https://github.com/xdas-dev/exercises.git
cd exercises
uv run install.py
```

That single command creates the environment, installs every library the
notebooks use, and downloads the dataset from
[Zenodo](https://doi.org/10.5281/zenodo.23533736) into `data/` (350 MB; see
`data/README.md` for its content). It is safe to re-run: a step already done
is skipped, and an interrupted download resumes where it stopped. Use
`uv run install.py --force` to fetch everything again.

Then start Jupyter — no environment to activate, `uv run` uses the right one:

```
uv run jupyter lab
```

## Real time

Notebook 4 needs `replay.py` running in a terminal. It plays the two cables
as if they were live, over ZeroMQ, as an ASN OptoDAS interrogator would,
stamped with the current UTC time:

```
uv run replay.py
```

| Stream | Address | Array |
|---|---|---|
| SER/N | `tcp://localhost:5556` | `(time, distance)`, 100 Hz, packets of 0.24 s |
| SER/S | `tcp://localhost:5557` | `(time, distance)`, 62.5 Hz, packets of 0.384 s |

The records loop every 3.5 min, so the earthquake comes back at every loop.
`--speedup 2` plays twice as fast as real time.

## Keeping up to date

To fetch the latest version of the exercises:

```
git pull
```

To reset the folder to its initial state (this does not touch `data/`):

```
git reset --hard HEAD
```

## License

The code is licensed under the GPL-3.0 (see `LICENSE.md`). The dataset is
under CC BY 4.0 (see `data/LICENSE.md`); the DAS records are provided by the
ABYSS project (ERC grant agreement No. 101041092).
