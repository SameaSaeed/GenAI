from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    username: str
    query: str
    previous_context: Optional[str] = None 