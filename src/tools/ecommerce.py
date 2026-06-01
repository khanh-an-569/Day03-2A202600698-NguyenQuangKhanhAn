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
    query: str
    items: List[CompetitorItem]
    summary: str