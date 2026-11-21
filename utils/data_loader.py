from io import BytesIO
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/587/basketball+dataset.zip"
ARCHIVE = Path(__file__).resolve().parents[1] / "data/raw/basketball.zip"
CHANNELS = ["x", "y", "z"]
RAW_COLUMNS = ["time_s", *CHANNELS, "reported_magnitude", "theta_deg", "phi_deg"]
TRIAL = re.compile(r"(\w+)_(dribble|hold|pass|pickup|shoot)(\d+)\.txt$")


def load_archive(path=ARCHIVE):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=60) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            if archive.testzip() is not None:
                raise ValueError("UCI archive failed its CRC check.")
        with NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as temporary:
            temporary.write(payload)
        pending = Path(temporary.name)
        try:
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    frames = []
    signals = {}
    with ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            name = PurePosixPath(member).name
            match = TRIAL.fullmatch(name)
            if not match and not name.startswith("Accelerometer Data "):
                continue
            payload = archive.read(member)
            header = payload.splitlines()[3].replace(b" ", b"")
            if header != b"Time(s),X(m/s2),Y(m/s2),Z(m/s2),R(m/s2),Theta(deg),Phi(deg)":
                raise ValueError(f"Unexpected sensor header in {member}")
            frame = pd.read_csv(BytesIO(payload), skiprows=4, header=None, names=RAW_COLUMNS)
            person, activity, trial = match.groups() if match else (None, "mixed", None)
            fingerprint = sha256(b"\n".join(payload.splitlines()[4:])).digest()
            first = signals.setdefault((person, activity, fingerprint), name)
            frame = frame.assign(recording=name, participant=person, activity=activity,
                                 trial=int(trial) if trial else None,
                                 recorded_at=payload.splitlines()[1].decode("ascii"),
                                 source=member, duplicate_of=first if first != name else None)
            frames.append(frame)
    if not frames:
        raise ValueError("No basketball recordings were found in the archive.")
    return pd.concat(frames, ignore_index=True)
