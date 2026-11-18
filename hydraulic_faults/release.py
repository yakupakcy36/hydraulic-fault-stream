"""Atomic, self-contained local releases. Hashes require a trusted publisher."""

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

from hydraulic_faults.contract import InputError, _unique_object
from hydraulic_faults.model import load_model_bytes
from hydraulic_faults.valve import load_valve_model_bytes

MAX_RELEASE_BYTES = 8_000_000


def encode(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path, limit: int) -> dict:
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise InputError("JSON file exceeds size limit")
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(data, dict):
            raise InputError("expected JSON object")
        return data
    except (ValueError, RecursionError) as exc:
        raise InputError("invalid JSON file") from exc


def atomic_json(path: Path, data: dict, *, replace: bool = False) -> None:
    raw = encode(data)
    if len(raw) > MAX_RELEASE_BYTES:
        raise InputError("release exceeds size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        if replace:
            os.replace(temporary, path)
        elif os.name == "nt":
            # Windows rename refuses an existing destination atomically.
            os.rename(temporary, path)
        else:
            # Atomically refuse overwrites, even with concurrent publishers.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def make_release(kind: str, model: dict, report: dict, provenance: dict) -> dict:
    payload = {"schema_version": 1, "kind": kind, "model": model, "report": report, "provenance": provenance}
    return {"release_id": digest(encode(payload)), "payload": payload}


def load_release(path: Path):
    data = read_json(path, MAX_RELEASE_BYTES)
    try:
        if data.keys() != {"release_id", "payload"}:
            raise InputError("invalid release envelope")
        payload = data["payload"]
        if data["release_id"] != digest(encode(payload)):
            raise InputError("release hash mismatch")
        if (
            payload.keys() != {"schema_version", "kind", "model", "report", "provenance"}
            or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
            or payload["kind"] not in ("rolling", "valve")
            or not isinstance(payload["report"], dict) or not isinstance(payload["provenance"], dict)
        ):
            raise InputError("invalid release schema")
        loader = load_model_bytes if payload["kind"] == "rolling" else load_valve_model_bytes
        model = loader(encode(payload["model"]))
    except (ValueError, TypeError, AttributeError, KeyError, RecursionError) as exc:
        raise InputError(f"invalid release: {exc}") from exc
    return data, model


def activate_release(release: Path, pointer: Path, config: Path) -> str:
    from hydraulic_faults.service import load_service_config

    release, config, pointer = release.resolve(), config.resolve(), pointer.resolve()
    if pointer in (release, config):
        raise InputError("active pointer must differ from release and configuration")
    if pointer.exists():
        load_active(pointer)
    bundle, _ = load_release(release)
    settings = load_service_config(config, bundle["payload"]["kind"])
    # Snapshot configuration in the pointer; rollback restores both together.
    atomic_json(pointer, {
        "schema_version": 1, "release": str(release), "release_id": bundle["release_id"],
        "config": settings, "config_id": digest(encode(settings)),
    }, replace=True)
    return bundle["release_id"]


def load_active(pointer: Path):
    data = read_json(pointer, 100_000)
    try:
        if (
            data.keys() != {"schema_version", "release", "release_id", "config", "config_id"}
            or type(data["schema_version"]) is not int or data["schema_version"] != 1
            or not isinstance(data["release"], str) or not Path(data["release"]).is_absolute()
            or data["config_id"] != digest(encode(data["config"]))
        ):
            raise InputError("invalid active pointer")
        bundle, model = load_release(Path(data["release"]))
        if data["release_id"] != bundle["release_id"]:
            raise InputError("active release identity mismatch")
        return bundle, model, data["config"]
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError(f"invalid active pointer: {exc}") from exc


def source_provenance(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], timeout=10).decode().strip()

    files = sorted((root / "hydraulic_faults").glob("*.py")) + [root / "pyproject.toml", root / "uv.lock"]
    if not files or not (root / "hydraulic_faults" / "cli.py").is_file():
        raise InputError("source root must contain the training project")
    hashes = {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in files}
    return {
        "git_revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
        "source_files": hashes, "source_id": digest(encode(hashes)),
        "versions": {name: version(name) for name in ("numpy", "scikit-learn")},
        "python": platform.python_version(),
    }


def train_release(archive: Path, output: Path, kind: str, source_root: Path) -> str:
    from zipfile import ZipFile

    from hydraulic_faults.dataset import MAX_ARCHIVE_BYTES, _profiles
    from hydraulic_faults.train import split_cycles, train_and_evaluate
    from hydraulic_faults.train_valve import train_valve

    source_root = source_root.resolve()
    # Prevent claiming provenance from a different checkout than the executing code.
    if Path(__file__).resolve().parent != source_root / "hydraulic_faults":
        raise InputError("source root must match the executing training code")
    provenance = source_provenance(source_root)
    if output.exists() or output.resolve() == archive.resolve():
        raise InputError("release output must be a new path distinct from the archive")
    with archive.open("rb") as data:
        if os.fstat(data.fileno()).st_size > MAX_ARCHIVE_BYTES:
            raise InputError("dataset exceeds size limit")
        initial_dataset_hash = hashlib.file_digest(data, "sha256").hexdigest()
    with ZipFile(archive) as data:
        train, test = split_cycles(_profiles(data))
    provenance["split"] = {"train": train.tolist(), "test": test.tolist()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        model_path = Path(staging) / "model.json"
        trainer = train_and_evaluate if kind == "rolling" else train_valve
        report = trainer(archive, model_path)
        with archive.open("rb") as data:
            if hashlib.file_digest(data, "sha256").hexdigest() != report["dataset_sha256"] or report["dataset_sha256"] != initial_dataset_hash:
                raise InputError("dataset changed during training")
        after = source_provenance(source_root)
        if after["source_id"] != provenance["source_id"] or after["git_revision"] != provenance["git_revision"]:
            raise InputError("training source changed during evaluation")
        report["evidence_scope"] = "development_reference_single_rig_not_external_validation"
        bundle = make_release(kind, read_json(model_path, 65_536), report, provenance)
        candidate = Path(staging) / "release.json"
        atomic_json(candidate, bundle)
        load_release(candidate)
        atomic_json(output, bundle)
    return bundle["release_id"]
