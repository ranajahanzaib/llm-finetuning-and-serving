# Architecture

LLM Fine-tuning and Serving owns Banking77 preparation, LoRA orchestration, task-specific inference, intent evaluation, the file-based run registry, and the local API/UI. The MLX-LM package owns optimization, and AfterTune owns paired capability grading and native regression reports.

The prediction path uses the model's chat template and requests a JSON BANKING77 intent. The intent is validated through Pydantic, then conservative policy rules supply advisory actions and review flags. Priority and entity fields use fallback defaults because the training data does not annotate them. Response metadata distinguishes model, policy, and fallback values. Sensitive activity, mixed issues, unknown intent, and instruction-like input trigger review; the service does not execute suggested actions.

The model manager loads lazily and serializes requests. Adapted inference requires a configured, compatible adapter and reports an error when one is unavailable; it never silently substitutes the base model. Comparison is sequential so two model variants need not be resident together.

Experiments and reports are local JSON files. No database or remote tracking service is required. MLX weights, downloaded data, run outputs, and adapters are ignored by Git.
