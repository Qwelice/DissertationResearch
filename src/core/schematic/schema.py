from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Type

from core.schematic.enums import SchemaType


@dataclass
class Schema:
    name: str
    schema_type: SchemaType
    params: Optional[Dict[str, Any]]
    dependencies: Optional[List[str]]
    strategy: Optional[str]
    output_type: Optional[Type]

    def link_strategy(self, strategy_name, output_type: Type):
        self.strategy = strategy_name
        self.output_type = output_type