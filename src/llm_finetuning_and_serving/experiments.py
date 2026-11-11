"""Small local JSON experiment registry."""

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4


def registry_root() -> Path:
    return Path("runs")


def create_run(base_model: str, adapter_path: str | None, config: dict, dataset: str) -> dict:
    run_id = f"run-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = registry_root() / run_id
    root.mkdir(parents=True)
    versions = {}
    for package in ("mlx", "mlx-lm", "datasets", "huggingface-hub", "pydantic"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = Path(dataset) / "manifest.json"
    dataset_manifest = json.loads(manifest.read_text()) if manifest.is_file() else {}
    import os

    record = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": base_model,
        "model_revision": os.getenv("LLM_FINETUNING_AND_SERVING_MODEL_REVISION"),
        "adapter_path": adapter_path,
        "dataset": dataset,
        "dataset_revision": dataset_manifest.get("dataset_revision"),
        "dataset_hashes": {
            name: values.get("sha256")
            for name, values in dataset_manifest.get("splits", {}).items()
        },
        "tokenizer_revision": os.getenv("LLM_FINETUNING_AND_SERVING_MODEL_REVISION"),
        "prompt_template_version": "banking-intent-v1",
        "training_config": config,
        "metrics": {},
        "status": "created",
        "artifacts": {},
        "git_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_versions": versions,
    }
    (root / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def list_runs(root: Path | None = None) -> list[dict]:
    root = root or registry_root()
    records = []
    if root.exists():
        for file in root.glob("run-*/manifest.json"):
            try:
                records.append(json.loads(file.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
    return sorted(records, key=lambda row: row.get("created_at", ""), reverse=True)


def get_run(run_id: str, root: Path | None = None) -> dict | None:
    if "/" in run_id or ".." in run_id:
        return None
    path = (root or registry_root()) / run_id / "manifest.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
