from datetime import datetime, timezone
from typing import Optional, List

from pydantic import BaseModel, field_validator


class CompetitorItem(BaseModel):
    product_name: str
    source: str
    price: Optional[float] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    url: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("rating")
    @classmethod
    def rating_in_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 5.0):
            raise ValueError(f"rating must be 0–5, got {v}")
        return v

    @field_validator("price")
    @classmethod
    def price_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError(f"price must be ≥ 0, got {v}")
        return v


class AgentMeta(BaseModel):
    """Runtime metadata — not part of the LLM output, added by the agent after validation."""
    timestamp: str = ""
    model: str = ""
    provider: str = ""
    steps: int = 0
    latency_ms: int = 0
    tool_calls: int = 0


class CompetitorReport(BaseModel):
    """
    Schema for the agent's final output.
    - items / query / summary  → filled by the LLM
    - meta                     → filled by ReActAgent after LLM validation
    """
    query: str
    items: List[CompetitorItem]
    summary: str
    meta: Optional[AgentMeta] = None

    @field_validator("items")
    @classmethod
    def items_not_empty(cls, v: List[CompetitorItem]) -> List[CompetitorItem]:
        if not v:
            raise ValueError("items must not be empty")
        return v