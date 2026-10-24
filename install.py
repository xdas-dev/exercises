"""Set up the exercises in one command:

    uv run install.py

`uv run` installs the dependencies listed in pyproject.toml before this script
starts; the script then downloads the dataset from Zenodo and unzips it into
`data/`: DAS records of the two SER cables, cable geometry, seismological
stations, topography and the earthquake. After it, the notebooks run offline.

It is skipped if the data is already there, and an interrupted download is
resumed where it stopped, so re-running is cheap. Pass --force to start over.
"""

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Report as the work happens, even when the output is piped to a file.
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).parent
DATA = ROOT / "data"

DATA_URL = "https://zenodo.org/records/23533736/files/data.zip"
DATA_MD5 = "e42e04bf35b2c0c1233bb7a2539585c0"
DATA_ZIP = ROOT / "data.zip"
# What the archive unpacks to. Used to tell "already downloaded" from "not yet".
DATA_DIRS = ("das", "geom", "seismo", "topo", "event")


def human(size):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def download(url, dest):
    """Download `url` to `dest`, resuming a partial file when the server allows."""
    done = dest.stat().st_size if dest.exists() else 0
    request = urllib.request.Request(url)
    if done:
        request.add_header("Range", f"bytes={done}-")

    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if error.code == 416:  # range not satisfiable: already complete
            return
        raise

    with response:
        if done and response.status != 206:  # no resume, start over
            done = 0
        total = int(response.headers.get("Content-Length", 0)) + done
        if done:
            print(f"resuming at {human(done)} of {human(total)}")
        else:
            print(f"downloading {human(total)} from {url}")

        start, resumed = time.monotonic(), done
        with open(dest, "ab" if done else "wb") as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)
                done += len(chunk)
                rate = (done - resumed) / max(time.monotonic() - start, 1e-6)
                percent = f"{100 * done / total:.1f}%, " if total else ""
                print(
                    f"\r  {human(done)} / {human(total) if total else '?'}"
                    f"  ({percent}{human(rate)}/s)",
                    end="",
                    flush=True,
                )
    print()

    if total and dest.stat().st_size != total:
        raise RuntimeError(
            f"{dest.name} is {human(dest.stat().st_size)}, expected {human(total)}. "
            "Re-run to resume."
        )


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_data(force):
    """Download and unzip the dataset."""
    if not force and all((DATA / name).is_dir() for name in DATA_DIRS):
        print(f"data: already in {DATA}, skipping")
        return

    if force and DATA_ZIP.exists():
        DATA_ZIP.unlink()

    download(DATA_URL, DATA_ZIP)

    print("checking the archive")
    if md5(DATA_ZIP) != DATA_MD5:
        # Most often a data.zip left over from something else: resuming onto it
        # produces a file of the right size whose contents do not line up.
        DATA_ZIP.unlink()
        raise RuntimeError(
            f"{DATA_ZIP.name} is corrupted, so it has been deleted. "
            "Re-run to download it again."
        )

    print(f"unzipping into {DATA}")
    with zipfile.ZipFile(DATA_ZIP) as archive:
        archive.extractall(ROOT)  # the archive holds a top-level data/ folder
    DATA_ZIP.unlink()

    missing = [name for name in DATA_DIRS if not (DATA / name).is_dir()]
    if missing:
        raise RuntimeError(f"the archive lacks {', '.join(missing)}")
    print(f"data: ready in {DATA}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force", action="store_true", help="download again even if already there"
    )
    args = parser.parse_args()

    fetch_data(args.force)

    print("\nready. Start the notebooks with:\n\n    uv run jupyter lab\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted -- re-run to pick up where it stopped")
        sys.exit(130)
