"""Metrics and transparent per-example evaluation reports."""

import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from llm_finetuning_and_serving.datasets import sha256
from llm_finetuning_and_serving.inference import InvalidModelOutput, ModelManager
from llm_finetuning_and_serving.prompts import PROMPT_VERSION


def metrics(expected: list[str], predicted: list[str], labels: list[str] | None = None) -> dict:
    if len(expected) != len(predicted) or not expected:
        raise ValueError("Expected and predicted labels must have the same nonzero length")
    labels = labels or sorted(set(expected) | set(predicted))
    if set(expected) - set(labels):
        raise ValueError("Expected labels contain values outside the supplied label registry")
    metric_labels = sorted(set(labels) | (set(predicted) - {"<invalid>"}))
    per_class = {}
    for label in metric_labels:
        tp = sum(a == label and b == label for a, b in zip(expected, predicted))
        fp = sum(a != label and b == label for a, b in zip(expected, predicted))
        fn = sum(a == label and b != label for a, b in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "support": sum(a == label for a in expected),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    confusion = {
        label: dict(Counter(b for a, b in zip(expected, predicted) if a == label))
        for label in labels
    }
    return {
        "n": len(expected),
        "accuracy": sum(a == b for a, b in zip(expected, predicted)) / len(expected),
        "macro_f1": sum(per_class[label]["f1"] for label in labels) / len(labels),
        "weighted_f1": sum(per_class[label]["f1"] * per_class[label]["support"] for label in labels)
        / len(expected),
        "per_intent": {label: per_class[label] for label in labels},
        "confusion_matrix": confusion,
    }


def evaluate(
    input_path: Path,
    output_path: Path,
    model: str,
    manager: ModelManager | None = None,
    few_shot_examples: list[dict] | None = None,
    labels: list[str] | None = None,
) -> dict:
    manager = manager or ModelManager()
    manager.prepare("base" if model == "few-shot" else model)
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    results = []
    expected_reviews = []
    actual_reviews = []
    for row in rows:
        started = time.perf_counter()
        try:
            response = manager.predict(
                row["text"], "base" if model == "few-shot" else model, few_shot_examples
            )
            intent = response.ticket.intent
            invalid = response.invalid_label
            if "requires_human_review" in row:
                expected_reviews.append(bool(row["requires_human_review"]))
                actual_reviews.append(response.ticket.requires_human_review)
            results.append(
                {
                    "input": row["text"],
                    "category": row.get("category"),
                    "expected": row["intent"],
                    "predicted": intent,
                    "expected_requires_human_review": row.get("requires_human_review"),
                    "predicted_requires_human_review": response.ticket.requires_human_review,
                    "passed": row["intent"] == intent and not invalid,
                    "json_valid": True,
                    "schema_valid": True,
                    "invalid_label": invalid,
                    "latency_ms": response.latency_ms,
                    "output_tokens": response.output_tokens,
                    "generation_tokens_per_second": response.generation_tokens_per_second,
                    "model_id": response.model_id,
                    "adapter_id": response.adapter_id,
                    "error_category": "invalid_label"
                    if invalid
                    else (None if row["intent"] == intent else "wrong_intent"),
                    "raw_output": response.raw_output,
                }
            )
        except InvalidModelOutput as exc:
            if "requires_human_review" in row:
                expected_reviews.append(bool(row["requires_human_review"]))
                actual_reviews.append(True)
            results.append(
                {
                    "input": row["text"],
                    "category": row.get("category"),
                    "expected": row["intent"],
                    "predicted": None,
                    "expected_requires_human_review": row.get("requires_human_review"),
                    "predicted_requires_human_review": True,
                    "passed": False,
                    "json_valid": False,
                    "schema_valid": False,
                    "invalid_label": False,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "output_tokens": None,
                    "generation_tokens_per_second": None,
                    "model_id": manager.model_id,
                    "adapter_id": manager.current["adapter_id"] if model == "adapted" else None,
                    "error_category": "invalid_json_or_schema",
                    "raw_output": exc.raw_output,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "input": row["text"],
                    "expected": row["intent"],
                    "predicted": None,
                    "passed": None,
                    "json_valid": None,
                    "schema_valid": None,
                    "invalid_label": False,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "output_tokens": None,
                    "generation_tokens_per_second": None,
                    "model_id": manager.model_id,
                    "adapter_id": manager.current["adapter_id"] if model == "adapted" else None,
                    "error_category": "inference_error",
                    "raw_output": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    generated = [x for x in results if x["error_category"] != "inference_error"]
    scored = [x for x in results if x["passed"] is not None]
    if scored:
        expected = [x["expected"] for x in scored]
        predicted = [x["predicted"] if x["predicted"] else "<invalid>" for x in scored]
        metric_labels = list(labels) if labels is not None else None
        if metric_labels is not None and "unknown" in expected and "unknown" not in metric_labels:
            metric_labels.append("unknown")
        score = metrics(expected, predicted, labels=metric_labels)
    else:
        score = {
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "per_intent": {},
            "confusion_matrix": {},
        }
    generated_count = len(generated)
    score.update(
        {
            "attempted": len(results),
            "scored": len(scored),
            "inference_errors": len(results) - generated_count,
            "json_valid": sum(x["json_valid"] is True for x in generated),
            "json_valid_rate": sum(x["json_valid"] is True for x in generated) / generated_count
            if generated_count
            else None,
            "schema_valid": sum(x["schema_valid"] is True for x in generated),
            "schema_valid_rate": sum(x["schema_valid"] is True for x in generated) / generated_count
            if generated_count
            else None,
            "invalid_label": sum(x["invalid_label"] for x in generated),
            "invalid_label_rate": sum(x["invalid_label"] for x in generated) / generated_count
            if generated_count
            else None,
        }
    )
    if expected_reviews:
        true_positive = sum(
            expected and predicted for expected, predicted in zip(expected_reviews, actual_reviews)
        )
        false_positive = sum(
            not expected and predicted
            for expected, predicted in zip(expected_reviews, actual_reviews)
        )
        false_negative = sum(
            expected and not predicted
            for expected, predicted in zip(expected_reviews, actual_reviews)
        )
        score["human_review"] = {
            "n": len(expected_reviews),
            "accuracy": sum(a == b for a, b in zip(expected_reviews, actual_reviews))
            / len(expected_reviews),
            "precision": true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0,
            "recall": true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0,
            "abstention_precision": true_positive / sum(actual_reviews)
            if sum(actual_reviews)
            else 0.0,
            "abstention_recall": true_positive / sum(expected_reviews)
            if sum(expected_reviews)
            else 0.0,
        }
    latencies = [x["latency_ms"] for x in generated]
    sorted_latencies = sorted(latencies)
    latency_summary = {
        "n": len(latencies),
        "median_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": sorted_latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
    }
    dataset_manifest = input_path.parent / "manifest.json"
    report = {
        "model_variant": model,
        "prompt_template_version": PROMPT_VERSION,
        "few_shot_strategy": "first four distinct training intents in stable BANKING77 order"
        if few_shot_examples
        else None,
        "few_shot_examples": few_shot_examples or [],
        "metrics": score,
        "examples": results,
        "evaluation_file": str(input_path.resolve()),
        "evaluation_sha256": sha256(input_path),
        "evaluation_examples": len(results),
        "adapter_id": manager.current["adapter_id"] if model == "adapted" else None,
        "model_id": manager.model_id,
        "model_revision": getattr(manager, "model_revision", None),
        "dataset_revision": json.loads(dataset_manifest.read_text()).get("dataset_revision")
        if dataset_manifest.is_file()
        else None,
        "latency": latency_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
