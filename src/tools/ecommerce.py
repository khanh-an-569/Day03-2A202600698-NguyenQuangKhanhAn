from pydantic import BaseModel, HttpUrl
from typing import Optional, List

class CompetitorItem(BaseModel):
    product_name: str
    source: str
    price: Optional[float] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    url: Optional[str] = None
    notes: Optional[str] = None

class CompetitorReport(BaseModel):
    timestamp: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    steps: Optional[int] = None
    latency_ms: Optional[int] = None
    tool_calls: Optional[int] = None
    observations_count: Optional[int] = None
    query: str
    items: List[CompetitorItem]
    summary: str