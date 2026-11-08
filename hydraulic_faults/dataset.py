from itertools import islice
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from hydraulic_faults.contract import CHANNELS, CONDITIONS, MAX_SENSOR_MAGNITUDE, InputError

CYCLE_SECONDS = 60
MAX_CYCLES = 2_205
MAX_ARCHIVE_BYTES = 100_000_000
MAX_ARCHIVE_MEMBER_BYTES = 110_000_000
MAX_SENSOR_ROW_BYTES = 250_000


def _rows(archive, member, columns, dtype=float):
    if archive.namelist().count(member) != 1:
        raise InputError(f"expected one {member} in the UCI ZIP")
    if archive.getinfo(member).file_size > MAX_ARCHIVE_MEMBER_BYTES:
        raise InputError(f"{member} exceeds size limit")
    with archive.open(member) as source:
        for row_number in range(1, MAX_CYCLES + 2):
            raw = source.readline(MAX_SENSOR_ROW_BYTES + 1)
            if not raw:
                break
            if len(raw) > MAX_SENSOR_ROW_BYTES or row_number > MAX_CYCLES:
                raise InputError(f"{member} exceeds row limit")
            values = np.fromstring(raw, dtype=dtype, sep="\t")
            if values.size != columns or not np.isfinite(values).all():
                raise InputError(f"invalid row {row_number} in {member}")
            yield values


def _profiles(archive):
    profiles = np.asarray(list(_rows(archive, "profile.txt", 5, dtype=int)))
    if not profiles.size:
        raise InputError("profile.txt is empty")
    for column, labels in enumerate(CONDITIONS.values()):
        if not np.isin(profiles[:, column], list(labels)).all():
            raise InputError("unknown condition in profile.txt")
    if not np.isin(profiles[:, 4], [0, 1]).all():
        raise InputError("unknown stability flag in profile.txt")
    return profiles


def load_cycles(path: Path, cycle: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read the allowlisted channels without extracting the ZIP to disk."""
    if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise InputError("expected a local UCI ZIP within the archive size limit")
    with ZipFile(path) as archive:
        profiles = _profiles(archive)
        total = len(profiles)
        if cycle is not None and not 0 <= cycle < total:
            raise InputError(f"cycle must be between 0 and {total - 1}")
        selected = total if cycle is None else 1
        samples = np.empty((selected, CYCLE_SECONDS, len(CHANNELS)), dtype=float)
        for column, channel in enumerate(CHANNELS):
            rows = _rows(archive, channel.member, CYCLE_SECONDS * channel.hz)
            try:
                selected_rows = rows if cycle is None else islice(rows, cycle, cycle + 1)
                count = 0
                for row in selected_rows:
                    if count >= selected:
                        raise InputError(f"unexpected cycle count in {channel.member}")
                    if ((row < -MAX_SENSOR_MAGNITUDE) | (row > MAX_SENSOR_MAGNITUDE)).any():
                        raise InputError(f"invalid sensor magnitude in {channel.member}")
                    samples[count, :, column] = row.reshape(CYCLE_SECONDS, channel.hz).mean(axis=1)
                    count += 1
                if count != selected:
                    raise InputError(f"unexpected cycle count in {channel.member}")
            finally:
                rows.close()
    return samples, profiles if cycle is None else profiles[cycle:cycle + 1]
