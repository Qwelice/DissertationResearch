from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Type

from core.schematic import SchemaType


@dataclass
class Schema:
    name: str
    schema_type: SchemaType
    params: Optional[Dict[str, Any]]
    dependencies: List[str]
    strategy: Optional[str]
    output_type: Optional[Type]