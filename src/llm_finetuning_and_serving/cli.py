"""LLM Fine-tuning and Serving command line interface."""

import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import typer

from llm_finetuning_and_serving import __version__
from llm_finetuning_and_serving.datasets import inspect, prepare
from llm_finetuning_and_serving.evaluation import evaluate
from llm_finetuning_and_serving.experiments import create_run, get_run, list_runs
from llm_finetuning_and_serving.inference import DEFAULT_MODEL, ModelManager
from llm_finetuning_and_serving.integrations.finetuning_regression_tester import (
    detect,
    load_report,
)
from llm_finetuning_and_serving.integrations.finetuning_regression_tester import (
    run as tester_run,
)
from llm_finetuning_and_serving.labels import BANKING77_LABELS

app = typer.Typer(
    help="Train, evaluate, and serve specialized language models locally.", no_args_is_help=True
)
data_app = typer.Typer(help="BANKING77 dataset operations", no_args_is_help=True)
regression_app = typer.Typer(
    help="finetuning-regression-tester (AfterTune) integration", no_args_is_help=True
)
exp_app = typer.Typer(help="Local experiment registry", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(regression_app, name="regression-tester")
app.add_typer(regression_app, name="aftertune")
app.add_typer(exp_app, name="experiments")


@app.command()
def doctor() -> None:
    mlx_message = "Requires macOS on Apple Silicon"
    mlx_ok = False
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_lm  # noqa: F401

            mlx_ok, mlx_message = True, "MLX-LM imported and Metal is available"
        except Exception as exc:
            mlx_message = f"MLX-LM unavailable: {type(exc).__name__}: {exc}"
    free_disk_gib = round(shutil.disk_usage(Path.cwd()).free / 1024**3, 2)
    model_cached = Path(os.getenv("LLM_FINETUNING_AND_SERVING_MODEL", DEFAULT_MODEL)).expanduser().exists()
    if not model_cached:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=os.getenv("LLM_FINETUNING_AND_SERVING_MODEL", DEFAULT_MODEL),
                revision=os.getenv("LLM_FINETUNING_AND_SERVING_MODEL_REVISION") or None,
                local_files_only=True,
            )
            model_cached = True
        except Exception:
            model_cached = False
    checks = {
        "llm-finetuning-and-serving": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "apple_silicon_mlx": {"available": mlx_ok, "detail": mlx_message},
        "free_disk_gib": free_disk_gib,
        "model_cached_locally": model_cached,
        "dataset_prepared": Path("data/banking77/manifest.json").exists(),
        "aftertune": detect().__dict__,
        "model": DEFAULT_MODEL,
    }
    typer.echo(json.dumps(checks, indent=2))


@app.command()
def benchmark(
    model: str = typer.Option("adapted", help="base or adapted"),
    text: str = typer.Option("My replacement card has not arrived."),
    repeats: int = typer.Option(10, min=2, max=100),
) -> None:
    if model not in {"base", "adapted"}:
        raise typer.BadParameter("model must be base or adapted")
    manager = ModelManager()
    if model == "adapted" and manager.adapter_status()["compatible"] is not True:
        raise typer.BadParameter(str(manager.adapter_status()["detail"]))
    cold = manager.predict(text, model)
    manager.predict(text, model)  # Explicit warm-up before steady-state samples.
    samples = [manager.predict(text, model) for _ in range(repeats)]
    latencies = sorted(x.latency_ms for x in samples)
    p95 = latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)]
    result = {
        "model_variant": model,
        "model_id": manager.model_id,
        "model_revision": manager.model_revision,
        "adapter_id": manager.current["adapter_id"] if model == "adapted" else None,
        "cold_start_latency_ms": cold.latency_ms,
        "warmup_count": 1,
        "warm_repeats": repeats,
        "warm_latency_median_ms": statistics.median(latencies),
        "warm_latency_p95_ms": p95,
        "generation_tokens_per_second_median": statistics.median(
            x.generation_tokens_per_second or 0 for x in samples
        ),
        "output_tokens_median": statistics.median(x.output_tokens for x in samples),
        "decoding": {"max_tokens": 96, "sampler": "greedy"},
        "note": "Single-input measurement; repeat under documented conditions for benchmark claims.",
    }
    typer.echo(json.dumps(result, indent=2))


@data_app.command("prepare")
def data_prepare(
    output: Path = typer.Option(Path("data/banking77")),
    seed: int = 42,
    validation_fraction: float = 0.1,
) -> None:
    typer.echo(json.dumps(prepare(output, seed, validation_fraction), indent=2))


@data_app.command("inspect")
def data_inspect(dataset: Path = typer.Option(Path("data/banking77"))) -> None:
    typer.echo(json.dumps(inspect(dataset), indent=2))


@app.command()
def train(
    dataset: Path = typer.Option(Path("data/banking77")),
    output: Path = typer.Option(Path("artifacts/adapters")),
    model: str | None = None,
    preset: str = typer.Option("development", help="development or standard"),
    iters: int | None = None,
    batch_size: int | None = None,
    config: Path | None = typer.Option(None, help="YAML training configuration"),
) -> None:
    configured = {}
    if config:
        import yaml

        try:
            configured = yaml.safe_load(config.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise typer.BadParameter(f"Cannot read training config: {exc}") from exc
        if not isinstance(configured, dict) or not isinstance(configured.get("training", {}), dict):
            raise typer.BadParameter("Config must contain a training mapping")
    training_values = configured.get("training", {})
    preset = training_values.get("preset", preset)
    if preset not in {"development", "standard"}:
        raise typer.BadParameter("preset must be development or standard")
    model = model or configured.get("model", DEFAULT_MODEL)
    model_revision = configured.get("model_revision") or os.getenv("LLM_FINETUNING_AND_SERVING_MODEL_REVISION")
    iterations = (
        iters
        if iters is not None
        else training_values.get("iters", 20 if preset == "development" else 160)
    )
    batch = batch_size if batch_size is not None else training_values.get("batch_size", 2)
    learning_rate = training_values.get("learning_rate", 1e-5)
    max_seq_length = training_values.get("max_seq_length", 512)
    seed = training_values.get("seed", 42)
    rank = training_values.get("rank", 8)
    scale = training_values.get("scale", 20.0)
    num_layers = training_values.get("num_layers", 8)
    if not (
        1 <= iterations <= 5000
        and 1 <= batch <= 64
        and 128 <= max_seq_length <= 32768
        and 1 <= num_layers <= 256
        and 1 <= rank <= 1024
        and scale > 0
        and 0 < learning_rate <= 1
        and seed >= 0
    ):
        raise typer.BadParameter(
            "Invalid training values; check iteration/batch bounds, sequence length, LoRA rank/scale, learning rate, layer count, and seed"
        )
    training_data = dataset / "mlx"
    if not (training_data / "train.jsonl").exists() or not (training_data / "valid.jsonl").exists():
        raise typer.BadParameter("Training data is missing; run llm-finetuning-and-serving data prepare first")
    if not shutil.which("mlx_lm.lora"):
        raise typer.BadParameter(
            "mlx_lm.lora command not found. Install LLM Fine-tuning and Serving with [mlx] on Apple Silicon."
        )
    training_model = model
    training_model_revision = model_revision
    if "/" in model and not Path(model).exists():
        try:
            from huggingface_hub import snapshot_download

            training_model = snapshot_download(repo_id=model, revision=model_revision)
            training_model_revision = Path(training_model).name
        except Exception as exc:
            raise typer.BadParameter(f"Could not resolve model checkpoint {model}: {exc}") from exc
    run = create_run(
        model,
        None,
        {
            "preset": preset,
            "iters": iterations,
            "batch_size": batch,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
            "seed": seed,
            "rank": rank,
            "scale": scale,
            "num_layers": num_layers,
            "source_config": str(config) if config else None,
        },
        str(dataset),
    )
    run["model_revision"] = training_model_revision
    run["tokenizer_revision"] = training_model_revision
    target = output / run["run_id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    training_config = {
        "model": training_model,
        "train": True,
        "fine_tune_type": "lora",
        "batch_size": batch,
        "iters": iterations,
        "learning_rate": learning_rate,
        "data": str(training_data),
        "adapter_path": str(target),
        "seed": seed,
        "max_seq_length": max_seq_length,
        "num_layers": num_layers,
        "lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0},
        "mask_prompt": True,
        "grad_checkpoint": True,
        "steps_per_report": 5,
        "steps_per_eval": 5,
        "save_every": 10,
    }
    manifest_path = Path("runs") / run["run_id"] / "manifest.json"
    run.update(status="running", adapter_path=str(target), training_config=training_config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(run, indent=2) + "\n")
    config_path = manifest_path.parent / "mlx-lora.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(training_config, sort_keys=False))
    run["mlx_config"] = str(config_path.resolve())
    manifest_path.write_text(json.dumps(run, indent=2) + "\n")
    command = ["mlx_lm.lora", "--config", str(config_path)]
    try:
        log = manifest_path.parent / "training.log"
        training_started = time.perf_counter()
        with log.open("w") as stream:
            proc = subprocess.run(
                command, text=True, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        adapter_file = target / "adapters.safetensors"
        adapter_smoke = None
        failure = None
        training_complete = proc.returncode == 0 and adapter_file.is_file()
        if proc.returncode == 0 and adapter_file.is_file():
            try:
                smoke_manager = ModelManager(model_id=model, adapter_path=str(target))
                smoke_manager.prepare("adapted")
                adapter_smoke = {"load_status": "passed", "inference_status": "pending"}
                smoke_prediction = smoke_manager.predict(
                    "My replacement card has not arrived.", "adapted"
                )
                adapter_smoke.update(
                    inference_status="passed",
                    model_id=smoke_prediction.model_id,
                    model_revision=smoke_prediction.model_revision,
                    adapter_id=smoke_prediction.adapter_id,
                    latency_ms=smoke_prediction.latency_ms,
                )
            except Exception as exc:
                failure = f"Adapter load/inference smoke check failed: {type(exc).__name__}: {exc}"
                if adapter_smoke is None:
                    adapter_smoke = {
                        "load_status": "failed",
                        "inference_status": "not_run",
                        "error": failure,
                    }
                else:
                    adapter_smoke.update(inference_status="failed", error=failure)
        completed = training_complete
        run.update(
            status="completed" if completed else "failed",
            deployable=bool(adapter_smoke and adapter_smoke.get("inference_status") == "passed"),
            training_log=str(log.resolve()),
            exit_code=proc.returncode,
            training_duration_seconds=round(time.perf_counter() - training_started, 3),
            adapter_file=str(adapter_file.resolve()) if adapter_file.exists() else None,
            adapter_size_bytes=adapter_file.stat().st_size if adapter_file.is_file() else None,
            adapter_smoke=adapter_smoke,
            failure=failure,
        )
        manifest_path.write_text(json.dumps(run, indent=2) + "\n")
        if not completed:
            typer.echo(
                f"Training failed or adapter artifact is missing; inspect {log}. {failure or ''}",
                err=True,
            )
            raise typer.Exit(code=proc.returncode or 1)
        if not run["deployable"]:
            typer.echo(
                f"Training completed, but adapter verification failed; run is not deployable. {failure or ''}",
                err=True,
            )
    except KeyboardInterrupt:
        run["status"] = "interrupted"
        manifest_path.write_text(json.dumps(run, indent=2) + "\n")
        raise
    except OSError as exc:
        run.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        manifest_path.write_text(json.dumps(run, indent=2) + "\n")
        raise typer.BadParameter(f"Could not start MLX-LM training: {exc}") from exc
    typer.echo(json.dumps(run, indent=2))


@app.command("eval")
def eval_cmd(
    dataset: Path = typer.Option(Path("data/banking77")),
    split: str = "test",
    input_file: Path | None = typer.Option(None, "--input", help="Custom labeled JSONL suite"),
    model: str = typer.Option("adapted", help="base, few-shot, or adapted"),
    output: Path = typer.Option(Path("runs/evaluation.json")),
    run_id: str | None = typer.Option(None, help="Associate this artifact with an experiment"),
) -> None:
    if split not in {"train", "validation", "test"}:
        raise typer.BadParameter("split must be train, validation, or test")
    path = input_file or dataset / ("valid.jsonl" if split == "validation" else f"{split}.jsonl")
    if not path.is_file():
        raise typer.BadParameter(f"Missing evaluation data: {path}")
    if model not in {"base", "few-shot", "adapted"}:
        raise typer.BadParameter("model must be base, few-shot, or adapted")
    record = get_run(run_id) if run_id else None
    if run_id and not record:
        raise typer.BadParameter("Experiment run_id was not found")
    if model == "adapted" and run_id:
        if record.get("status") != "completed":
            raise typer.BadParameter("Adapted evaluation requires a completed training run")
        configured_adapter = os.getenv("LLM_FINETUNING_AND_SERVING_ADAPTER")
        if (
            not configured_adapter
            or Path(configured_adapter).resolve() != Path(record["adapter_path"]).resolve()
        ):
            raise typer.BadParameter(
                "LLM_FINETUNING_AND_SERVING_ADAPTER must identify the selected experiment adapter"
            )
    shots = []
    if model == "few-shot":
        training_rows = [
            json.loads(line)
            for line in (dataset / "train.jsonl").read_text().splitlines()
            if line.strip()
        ]
        seen = set()
        for row in training_rows:
            if row["intent"] not in seen:
                shots.append(row)
                seen.add(row["intent"])
            if len(shots) == 4:
                break
        if len(shots) != 4:
            raise typer.BadParameter("Few-shot evaluation needs four distinct training intents")
    report = evaluate(path, output, model, few_shot_examples=shots, labels=list(BANKING77_LABELS))
    if run_id:
        record.setdefault("artifacts", {})[f"evaluation_{model}_{split}"] = str(output.resolve())
        record["evaluation_metadata"] = {
            "model_variant": model,
            "split": split,
            "examples": report["evaluation_examples"],
            "metrics": report["metrics"],
        }
        (Path("runs") / run_id / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(json.dumps(report["metrics"], indent=2))


@app.command("compare")
def compare_cmd(
    base: Path = typer.Option(..., help="Base-model evaluation report JSON"),
    adapted: Path = typer.Option(..., help="Adapted-model evaluation report JSON"),
    output: Path = typer.Option(Path("runs/comparison.json")),
    run_id: str | None = typer.Option(None, help="Associate the comparison with an experiment"),
) -> None:
    try:
        base_report = json.loads(base.read_text())
        adapted_report = json.loads(adapted.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Cannot read evaluation reports: {exc}") from exc
    if (
        base_report.get("model_variant") != "base"
        or adapted_report.get("model_variant") != "adapted"
    ):
        raise typer.BadParameter("Comparison requires one base and one adapted evaluation report")
    if not adapted_report.get("adapter_id"):
        raise typer.BadParameter("The adapted report has no adapter provenance")
    fields = (
        "evaluation_file",
        "evaluation_sha256",
        "prompt_template_version",
        "evaluation_examples",
        "model_revision",
        "dataset_revision",
    )
    if any(base_report.get(key) != adapted_report.get(key) for key in fields):
        raise typer.BadParameter("Reports do not share the same evaluation set and prompt template")
    base_examples, adapted_examples = (
        base_report.get("examples", []),
        adapted_report.get("examples", []),
    )
    if [(x.get("input"), x.get("expected")) for x in base_examples] != [
        (x.get("input"), x.get("expected")) for x in adapted_examples
    ]:
        raise typer.BadParameter(
            "Per-example inputs or expected labels differ; paired comparison is invalid"
        )
    if base_report.get("model_id") != adapted_report.get("model_id"):
        raise typer.BadParameter("Base and adapted reports use different checkpoints")
    if (
        base_report.get("metrics", {}).get("accuracy") is None
        or adapted_report.get("metrics", {}).get("accuracy") is None
    ):
        raise typer.BadParameter(
            "Both reports need at least one scored model response for a paired comparison"
        )
    summary = {
        "n": base_report["metrics"]["n"],
        "base": base_report["metrics"],
        "adapted": adapted_report["metrics"],
        "accuracy_delta": adapted_report["metrics"]["accuracy"]
        - base_report["metrics"]["accuracy"],
        "macro_f1_delta": adapted_report["metrics"]["macro_f1"]
        - base_report["metrics"]["macro_f1"],
    }
    payload = {
        "schema_version": "1",
        "compatible": True,
        "evaluation_file": base_report["evaluation_file"],
        "prompt_template_version": base_report["prompt_template_version"],
        "model_id": base_report["model_id"],
        "adapter_id": adapted_report["adapter_id"],
        "base_report": str(base.resolve()),
        "adapted_report": str(adapted.resolve()),
        "summary": summary,
        "examples": [
            {
                "input": b["input"],
                "expected": b["expected"],
                "base_predicted": b["predicted"],
                "adapted_predicted": a["predicted"],
                "base_passed": b["passed"],
                "adapted_passed": a["passed"],
                "transition": "inconclusive"
                if b["passed"] is None or a["passed"] is None
                else "improvement"
                if not b["passed"] and a["passed"]
                else "regression"
                if b["passed"] and not a["passed"]
                else "preserved_success"
                if b["passed"]
                else "preserved_failure",
            }
            for b, a in zip(base_examples, adapted_examples)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    if run_id:
        record = get_run(run_id)
        if not record:
            raise typer.BadParameter("Experiment run_id was not found")
        if record.get("adapter_path") != adapted_report.get("adapter_id"):
            raise typer.BadParameter("Adapted report does not match the experiment adapter")
        record.setdefault("artifacts", {})["comparison"] = str(output.resolve())
        (Path("runs") / run_id / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(json.dumps(summary, indent=2))


@regression_app.command("check")
def tester_check() -> None:
    status = detect()
    typer.echo(json.dumps(status.__dict__, indent=2))
    if not status.available:
        raise typer.Exit(1)


@regression_app.command("run")
def tester_eval(
    run_id: str,
    cases: Path = typer.Option(
        ..., help="finetuning-regression-tester compatible JSONL evaluation suite"
    ),
    adapter: Path = typer.Option(...),
    config: Path = typer.Option(...),
    output: Path = typer.Option(...),
    timeout: int = 3600,
) -> None:
    record = get_run(run_id)
    if not record:
        raise typer.BadParameter("Unknown LLM Fine-tuning and Serving experiment run_id")
    if record.get("status") != "completed":
        raise typer.BadParameter(
            "finetuning-regression-tester can only evaluate a completed LLM Fine-tuning and Serving run"
        )
    if record.get("adapter_path") and Path(record["adapter_path"]).resolve() != adapter.resolve():
        raise typer.BadParameter("Adapter path does not match the selected LLM Fine-tuning and Serving experiment")
    if not adapter.is_dir() or not (adapter / "adapters.safetensors").is_file():
        raise typer.BadParameter("Adapter is missing its MLX-LM adapters.safetensors artifact")
    if not cases.is_file() or not config.is_file():
        raise typer.BadParameter("finetuning-regression-tester cases and config must both exist")
    import yaml

    config_data = yaml.safe_load(config.read_text()) or {}
    configured_model = config_data.get("model", {})
    configured_model = (
        configured_model.get("id") if isinstance(configured_model, dict) else configured_model
    )
    if configured_model != record.get("base_model"):
        raise typer.BadParameter(
            "finetuning-regression-tester config model must match the LLM Fine-tuning and Serving run's base model"
        )
    configured_revision = (
        config_data.get("model", {}).get("revision")
        if isinstance(config_data.get("model"), dict)
        else None
    )
    if record.get("model_revision") and configured_revision != record["model_revision"]:
        raise typer.BadParameter(
            "finetuning-regression-tester config must pin the same model revision as the LLM Fine-tuning and Serving run"
        )
    output.mkdir(parents=True, exist_ok=True)
    commands = [
        (
            "baseline",
            [
                "baseline",
                "--cases",
                str(cases),
                "--output",
                str(output / "baseline.jsonl"),
                "--config",
                str(config),
            ],
        ),
        (
            "evaluate",
            [
                "evaluate",
                "--cases",
                str(cases),
                "--adapter",
                str(adapter),
                "--output",
                str(output / "adapted.jsonl"),
                "--config",
                str(config),
            ],
        ),
        (
            "compare",
            [
                "compare",
                "--baseline",
                str(output / "baseline.jsonl"),
                "--adapted",
                str(output / "adapted.jsonl"),
                "--output",
                str(output / "comparison.json"),
            ],
        ),
        (
            "report",
            [
                "report",
                "--comparison",
                str(output / "comparison.json"),
                "--output",
                str(output / "report"),
            ],
        ),
    ]
    provenance = {
        "llm_finetuning_and_serving_run_id": run_id,
        "cases": str(cases.resolve()),
        "adapter": str(adapter.resolve()),
        "config": str(config.resolve()),
        "commands": [],
    }
    for stage, args in commands:
        result = tester_run(args, timeout)
        provenance["commands"].append(result)
        (output / "invocation.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if result["returncode"]:
            raise typer.BadParameter(
                f"finetuning-regression-tester {stage} failed: {result['stderr'][-2000:]}"
            )
    imported = load_report(output / "comparison.json", run_id)
    imported["invocation"] = provenance
    target = output / "llm-finetuning-and-serving-aftertune.json"
    target.write_text(json.dumps(imported, indent=2) + "\n")
    record.setdefault("artifacts", {})["finetuning_regression_tester"] = str(target.resolve())
    record.setdefault("artifacts", {})["aftertune"] = str(target.resolve())
    (Path("runs") / run_id / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(
        json.dumps(
            {"status": "completed", "report": str(target), "summary": imported["summary"]}, indent=2
        )
    )


@exp_app.command("list")
def experiments_list() -> None:
    typer.echo(json.dumps(list_runs(), indent=2))


@exp_app.command("show")
def experiments_show(run_id: str) -> None:
    record = get_run(run_id)
    if not record:
        raise typer.BadParameter("Experiment not found")
    typer.echo(json.dumps(record, indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("llm_finetuning_and_serving.api:app", host=host, port=port, reload=False)
