"""Model loading and conservative ticket inference."""

import json
import os
import time
from pathlib import Path
from threading import RLock

from llm_finetuning_and_serving.labels import BANKING77_LABELS
from llm_finetuning_and_serving.prompts import INTENT_SYSTEM_PROMPT
from llm_finetuning_and_serving.schemas import PredictionResponse, Ticket

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
POLICY_ACTIONS = {
    "cash_withdrawal_not_recognised": "review_cash_withdrawal_issue",
    "card_arrival": "check_card_delivery_status",
    "compromised_card": "review_suspected_compromise",
    "lost_or_stolen_card": "review_lost_or_stolen_card",
    "failed_transfer": "review_failed_transfer",
    "transfer_not_received_by_recipient": "review_transfer_trace",
    "transaction_charged_twice": "review_duplicate_transaction",
    "card_payment_not_recognised": "review_unrecognized_card_payment",
    "direct_debit_payment_not_recognised": "review_unrecognized_direct_debit",
}


def apply_policy(ticket: Ticket, text: str) -> tuple[Ticket, dict[str, str], list[str]]:
    """Enforce safety rules and identify deterministic versus model-sourced fields."""
    sources = {
        "intent": "model",
        "priority": "validation_fallback",
        "entities": "validation_fallback",
        "suggested_action": "deterministic_policy",
        "requires_human_review": "deterministic_policy",
    }
    warnings: list[str] = []
    normalized = text.casefold()
    risky = any(
        term in normalized
        for term in (
            "stolen",
            "fraud",
            "compromised",
            "unauthorized",
            "don't recognize",
            "do not recognize",
        )
    )
    multiple = any(word in normalized for word in (" and also ", " as well as ", " additionally "))
    injection = any(
        term in normalized
        for term in (
            "ignore your instructions",
            "reveal your prompt",
            "return a successful transfer",
        )
    )
    unrelated = any(
        term in normalized for term in ("weather tomorrow", "bake a cake", "capital of france")
    )
    if risky or multiple or injection or unrelated or ticket.intent == "unknown":
        ticket.requires_human_review = True
        sources["requires_human_review"] = "deterministic_policy"
    if injection or unrelated:
        ticket.intent = "unknown"
        ticket.suggested_action = None
        ticket.entities = {}
        ticket.requires_human_review = True
        sources.update(
            intent="deterministic_policy",
            suggested_action="deterministic_policy",
            entities="validation_fallback",
        )
        warnings.append("Input was treated as untrusted or outside the supported banking task.")
    if risky:
        warnings.append("Potentially sensitive or fraudulent activity requires human review.")
    if multiple:
        warnings.append("Multiple issues were detected; human review is required.")
    approved_action = load_policy_actions().get(ticket.intent)
    if ticket.suggested_action != approved_action:
        ticket.suggested_action = approved_action
        sources["suggested_action"] = (
            "deterministic_policy" if approved_action else "validation_fallback"
        )
    return ticket, sources, warnings


def load_policy_actions() -> dict[str, str]:
    """Load advisory action mappings; reject unknown labels and unsafe action identifiers."""
    policy_path = Path(
        os.getenv(
            "LLM_FINETUNING_AND_SERVING_POLICY",
            str(Path(__file__).resolve().parents[2] / "configs" / "policy.yaml"),
        )
    )
    if not policy_path.is_file():
        return POLICY_ACTIONS.copy()
    try:
        import yaml

        document = yaml.safe_load(policy_path.read_text()) or {}
    except Exception as exc:
        raise RuntimeError(f"Could not load policy config {policy_path}: {exc}") from exc
    actions = document.get("suggested_actions", {}) if isinstance(document, dict) else None
    if not isinstance(actions, dict):
        raise RuntimeError("Policy config must contain a suggested_actions mapping")
    validated = {}
    for intent, action in actions.items():
        if intent not in BANKING77_LABELS:
            raise RuntimeError(f"Policy action references unknown BANKING77 intent: {intent}")
        if (
            not isinstance(action, str)
            or not action.startswith(("review_", "check_", "route_"))
            or not action.replace("_", "").isalnum()
        ):
            raise RuntimeError(
                f"Policy action is not an approved advisory workflow identifier: {action}"
            )
        validated[intent] = action
    return validated


class ModelManager:
    """Lazy, serialized MLX model manager. Missing MLX is an explicit runtime error."""

    def __init__(self, model_id: str | None = None, adapter_path: str | None = None):
        self.model_id = model_id or os.getenv("LLM_FINETUNING_AND_SERVING_MODEL", DEFAULT_MODEL)
        self.model_revision = os.getenv("LLM_FINETUNING_AND_SERVING_MODEL_REVISION") or None
        self.adapter_path = adapter_path or os.getenv("LLM_FINETUNING_AND_SERVING_ADAPTER")
        if self.adapter_path and not self.model_revision:
            try:
                adapter_config = json.loads(
                    (Path(self.adapter_path) / "adapter_config.json").read_text()
                )
                trained_path = Path(str(adapter_config.get("model", "")))
                snapshot_revision = (
                    trained_path.name if trained_path.parent.name == "snapshots" else None
                )
                cache_repo = trained_path.parent.parent.name if snapshot_revision else ""
                expected_cache_repo = "models--" + self.model_id.replace("/", "--")
                if snapshot_revision and cache_repo == expected_cache_repo:
                    self.model_revision = snapshot_revision
            except (OSError, json.JSONDecodeError):
                pass
        self._loaded: dict[bool, tuple[object, object]] = {}
        self._lock = RLock()

    @property
    def current(self) -> dict[str, str | None]:
        return {
            "base_model": self.model_id,
            "adapter_id": str(self.adapter_path) if self.adapter_path else None,
        }

    def adapter_status(self) -> dict[str, str | bool | None]:
        if not self.adapter_path:
            return {"state": "unavailable", "compatible": None, "detail": "No adapter configured"}
        path = Path(self.adapter_path)
        if not path.is_dir() or not (path / "adapters.safetensors").is_file():
            return {
                "state": "incompatible",
                "compatible": False,
                "detail": "Adapter directory or adapters.safetensors is missing",
            }
        config_file = path / "adapter_config.json"
        if not config_file.is_file():
            return {
                "state": "incompatible",
                "compatible": False,
                "detail": "MLX-LM adapter_config.json is missing",
            }
        try:
            config = json.loads(config_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "state": "incompatible",
                "compatible": False,
                "detail": f"Cannot read adapter config: {exc}",
            }
        trained_model = config.get("model")
        trained_revision_matches = (
            self.model_revision and Path(str(trained_model)).name == self.model_revision
        )
        if trained_model != self.model_id and not trained_revision_matches:
            return {
                "state": "incompatible",
                "compatible": False,
                "detail": f"Adapter model {trained_model!r} does not match {self.model_id!r}",
            }
        return {
            "state": "compatible",
            "compatible": True,
            "detail": "Adapter metadata matches the selected base model",
        }

    def _load(self, adapted: bool) -> tuple[object, object]:
        key = adapted and bool(self.adapter_path)
        if key not in self._loaded:
            try:
                from mlx_lm import load
            except ImportError as exc:
                raise RuntimeError(
                    "MLX-LM is unavailable; install LLM Fine-tuning and Serving with the [mlx] extra on Apple Silicon."
                ) from exc
            if self._loaded:
                self._loaded.clear()
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except (ImportError, AttributeError):
                    pass
            if key and self.adapter_status()["compatible"] is not True:
                raise RuntimeError(str(self.adapter_status()["detail"]))
            args = {"adapter_path": self.adapter_path} if key else {}
            model_path = self.model_id
            if self.model_revision and "/" in model_path:
                from huggingface_hub import snapshot_download

                model_path = snapshot_download(repo_id=model_path, revision=self.model_revision)
            self._loaded[key] = load(model_path, **args)
        return self._loaded[key]

    def prepare(self, model: str = "adapted") -> None:
        """Fail once, before starting a potentially long evaluation loop, if loading fails."""
        with self._lock:
            if model == "adapted" and self.adapter_status()["compatible"] is not True:
                raise RuntimeError(str(self.adapter_status()["detail"]))
            self._load(model == "adapted")

    def predict(
        self, text: str, model: str = "adapted", few_shot_examples: list[dict] | None = None
    ) -> PredictionResponse:
        if model == "adapted" and self.adapter_status()["compatible"] is not True:
            raise RuntimeError(str(self.adapter_status()["detail"]))
        from mlx_lm import generate

        with self._lock:
            started = time.perf_counter()
            adapted = model == "adapted"
            tokenizer_model = self._load(adapted)
            model_object, tokenizer = tokenizer_model
            messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]
            if few_shot_examples:
                for example in few_shot_examples:
                    messages.extend(
                        [
                            {"role": "user", "content": example["text"]},
                            {
                                "role": "assistant",
                                "content": json.dumps({"intent": example["intent"]}),
                            },
                        ]
                    )
            messages.append({"role": "user", "content": text})
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            generation_started = time.perf_counter()
            raw = generate(model_object, tokenizer, prompt=prompt, max_tokens=96, verbose=False)
            generation_seconds = time.perf_counter() - generation_started
            output_tokens = len(tokenizer.encode(raw))
            try:
                parsed = json.loads(raw.strip())
                if (
                    not isinstance(parsed, dict)
                    or set(parsed) != {"intent"}
                    or not isinstance(parsed.get("intent"), str)
                ):
                    raise ValueError("expected a JSON object with only a string intent field")
                raw_intent = parsed["intent"]
                invalid_label = raw_intent not in BANKING77_LABELS and raw_intent != "unknown"
                ticket = Ticket(
                    intent=raw_intent if not invalid_label else "unknown",
                    requires_human_review=invalid_label,
                )
            except Exception as exc:
                raise InvalidModelOutput(raw, str(exc)) from exc
            ticket, sources, warnings = apply_policy(ticket, text)
            if invalid_label:
                sources["intent"] = "validation_fallback"
                ticket.requires_human_review = True
                warnings.append(
                    "The model returned an intent outside the BANKING77 label registry; mapped to unknown for review."
                )
            warnings.append(
                "BANKING77 provides intent labels only; priority and entities use conservative defaults until reviewed ticket annotations are available."
            )
            elapsed = (time.perf_counter() - started) * 1000
            return PredictionResponse(
                ticket=ticket,
                model_id=self.model_id,
                adapter_id=str(self.adapter_path) if adapted else None,
                model_revision=self.model_revision,
                invalid_label=invalid_label,
                latency_ms=round(elapsed, 2),
                raw_output=raw,
                output_tokens=output_tokens,
                generation_tokens_per_second=round(output_tokens / generation_seconds, 2)
                if generation_seconds > 0
                else None,
                warnings=warnings,
                field_sources=sources,
            )


class InvalidModelOutput(RuntimeError):
    """Model generation could not be parsed into the intent response contract."""

    def __init__(self, raw_output: str, reason: str):
        super().__init__(reason)
        self.raw_output = raw_output
