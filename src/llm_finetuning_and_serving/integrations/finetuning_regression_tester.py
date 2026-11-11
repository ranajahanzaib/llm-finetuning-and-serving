"""Thin subprocess adapter for finetuning-regression-tester's documented CLI and native reports."""

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RegressionTesterStatus:
    available: bool
    executable: str | None
    version: str | None
    detail: str


# Backward compatibility alias
AfterTuneStatus = RegressionTesterStatus


def detect(executable: str | None = None) -> RegressionTesterStatus:
    configured = (
        executable
        or os.getenv("FINETUNING_REGRESSION_TESTER_BIN")
        or os.getenv("AFTERTUNE_BIN")
    )
    path = (
        shutil.which(configured)
        if configured
        else (
            shutil.which("finetuning-regression-tester")
            or shutil.which("aftertune")
        )
    )
    if not path:
        repository = Path(__file__).resolve().parents[3]
        for candidate_name in ("finetuning-regression-tester", "aftertune"):
            sibling = repository.parent / candidate_name / ".venv" / "bin" / "finetuning-regression-tester"
            if sibling.is_file():
                path = str(sibling)
                break
            sibling_legacy = repository.parent / candidate_name / ".venv" / "bin" / "aftertune"
            if sibling_legacy.is_file():
                path = str(sibling_legacy)
                break
    if not path:
        return RegressionTesterStatus(False, None, None, "finetuning-regression-tester CLI not found")
    try:
        result = subprocess.run(
            [path, "--help"], capture_output=True, text=True, timeout=10, check=False
        )
        if result.returncode:
            return RegressionTesterStatus(False, path, None, result.stderr.strip() or "CLI check failed")
        ver = subprocess.run(
            [path, "doctor"], capture_output=True, text=True, timeout=10, check=False
        )
        first_line = next(
            (
                line
                for line in ver.stdout.splitlines()
                if line.startswith("FineTuning Regression Tester ") or line.startswith("AfterTune ")
            ),
            None,
        )
        version_str = "unknown"
        if first_line:
            if first_line.startswith("FineTuning Regression Tester "):
                version_str = first_line.removeprefix("FineTuning Regression Tester ")
            elif first_line.startswith("AfterTune "):
                version_str = first_line.removeprefix("AfterTune ")

        return RegressionTesterStatus(
            True,
            path,
            version_str,
            "CLI available",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return RegressionTesterStatus(False, path, None, str(exc))


def run(args: list[str], timeout: int = 3600, executable: str | None = None) -> dict:
    status = detect(executable)
    if not status.available or not status.executable:
        raise RuntimeError(status.detail)
    command = [status.executable, *args]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "command": command,
            "version": status.version,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout or ""
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr or ""
        )
        return {
            "command": command,
            "version": status.version,
            "returncode": -1,
            "stdout": stdout,
            "stderr": stderr + f"\nTimed out after {timeout} seconds",
            "timed_out": True,
        }
    except OSError as exc:
        return {
            "command": command,
            "version": status.version,
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "timed_out": False,
        }


def load_report(path: Path, run_id: str) -> dict:
    """Validate native finetuning-regression-tester v1 comparison and preserve source report unchanged."""
    report = json.loads(path.read_text())
    if report.get("schema_version") != "1" or report.get("compatible") is not True:
        raise ValueError("Unsupported or incompatible finetuning-regression-tester comparison report")
    if not isinstance(report.get("capabilities"), dict) or not isinstance(
        report.get("cases"), list
    ):
        raise ValueError("finetuning-regression-tester report lacks capability metrics or case evidence")
    return {
        "source": "finetuning-regression-tester",
        "source_report": str(path.resolve()),
        "llm_finetuning_and_serving_run_id": run_id,
        "imported_report": report,
        "summary": report.get("summary"),
        "capabilities": report["capabilities"],
        "cases": report["cases"],
    }


def status_dict() -> dict:
    return asdict(detect())
