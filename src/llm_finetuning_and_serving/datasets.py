"""BANKING77 acquisition, validation, and MLX chat conversion."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from llm_finetuning_and_serving.labels import BANKING77_REVISION
from llm_finetuning_and_serving.prompts import INTENT_SYSTEM_PROMPT

DATASET_ID = "PolyAI/banking77"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(output: Path, seed: int = 42, validation_fraction: float = 0.1) -> dict:
    """Download official split and stratify only its training partition."""
    if not 0 < validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be greater than 0 and no more than 0.5")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install LLM Fine-tuning and Serving with the [data] extra to download BANKING77.") from exc
    dataset = load_dataset(DATASET_ID, revision=BANKING77_REVISION)
    train, official_test = dataset["train"], dataset["test"]
    label_names = train.features["label"].names
    groups: dict[int, list[dict]] = {}
    for row in train:
        groups.setdefault(int(row["label"]), []).append(row)
    rng = random.Random(seed)
    fit, valid = [], []
    for label, rows in sorted(groups.items()):
        rng.shuffle(rows)
        cut = max(1, round(len(rows) * validation_fraction))
        valid.extend(rows[:cut])
        fit.extend(rows[cut:])
    output.mkdir(parents=True, exist_ok=True)
    mlx_dir = output / "mlx"
    mlx_dir.mkdir(exist_ok=True)
    records = {
        "train.jsonl": fit,
        "valid.jsonl": valid,
        "test.jsonl": list(official_test),
    }
    counts = {}
    normalized_splits = {}
    lengths = []
    for name, rows in records.items():
        path = output / name
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                record = {"text": row["text"], "intent": label_names[int(row["label"])]}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[name] = {"examples": len(rows), "sha256": sha256(path)}
        normalized_splits[name] = {" ".join(str(row["text"]).casefold().split()) for row in rows}
        lengths.extend(len(str(row["text"]).split()) for row in rows)
    duplicate_counts = {
        name: len(rows) - len({" ".join(str(row["text"]).casefold().split()) for row in rows})
        for name, rows in records.items()
    }
    overlap_counts = {}
    names = list(records)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap_counts[f"{left}:{right}"] = len(
                normalized_splits[left] & normalized_splits[right]
            )
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": BANKING77_REVISION,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "labels": label_names,
        "label_count": len(label_names),
        "splits": counts,
        "test_split_source": "official test; never used for training or prompt selection",
        "intent_counts_official_train": dict(Counter(label_names[int(r["label"])] for r in train)),
        "intent_counts_fit": dict(Counter(label_names[int(r["label"])] for r in fit)),
        "quality": {
            "exact_normalized_duplicates_by_split": duplicate_counts,
            "normalized_overlap_between_splits": overlap_counts,
            "whitespace_token_length": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": round(sum(lengths) / len(lengths), 2),
            },
            "note": "Token lengths are whitespace word counts, not tokenizer token counts. Overlaps are reported and retained.",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for split, rows in records.items():
        if split == "test.jsonl":
            continue
        chat_path = mlx_dir / split
        with chat_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                label = label_names[int(row["label"])]
                item = {
                    "messages": [
                        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                        {"role": "user", "content": row["text"]},
                        {"role": "assistant", "content": json.dumps({"intent": label})},
                    ]
                }
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return manifest


def inspect(path: Path) -> dict:
    """Summarize already prepared JSONL data without network access."""
    splits = {}
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        file = path / name
        if not file.exists():
            continue
        rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        splits[name] = {
            "examples": len(rows),
            "intent_counts": dict(Counter(row["intent"] for row in rows)),
            "sha256": sha256(file),
        }
    if not splits:
        raise FileNotFoundError(f"No prepared split files in {path}")
    return {
        "splits": splits,
        "manifest": json.loads((path / "manifest.json").read_text())
        if (path / "manifest.json").exists()
        else None,
    }
