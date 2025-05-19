from dataclasses import dataclass
from typing import List, Callable, Any, Dict, Optional

from core.schemas import SchemaType


@dataclass
class Schema:
    name: str
    type: SchemaType
    params: Optional[Dict[str, Any]]
    dependencies: List[str]
    strategy: Callable[..., Any]