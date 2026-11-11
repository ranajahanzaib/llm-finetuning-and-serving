"""Validated task inputs and outputs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str = Field(min_length=1, description="BANKING77 label or unknown")
    priority: Literal["low", "normal", "high", "unspecified"] = "unspecified"
    entities: dict[str, str] = Field(default_factory=dict)
    suggested_action: str | None = None
    requires_human_review: bool = False


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    model: Literal["base", "adapted"] = "adapted"


class PredictionResponse(BaseModel):
    ticket: Ticket
    model_id: str
    adapter_id: str | None = None
    model_revision: str | None = None
    invalid_label: bool = False
    raw_output: str
    output_tokens: int
    generation_tokens_per_second: float | None = None
    latency_ms: float
    validation_status: Literal["valid"] = "valid"
    warnings: list[str] = Field(default_factory=list)
    field_sources: dict[str, str] = Field(default_factory=dict)
