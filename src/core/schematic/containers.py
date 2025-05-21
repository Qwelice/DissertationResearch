from typing import Dict, Optional, Any, List, Callable

from core.schematic import Schema, SchemaStrategy
from core.schematic.enums import SchemaType


class SchemaContainer:
    def __init__(self, schema_type: SchemaType):
        self._schema_type = schema_type
        self._schemas: Dict[str, Schema] = {}

    @property
    def entry_count(self) -> int:
        return len(self._schemas)

    @property
    def is_empty(self) -> bool:
        return self.entry_count < 1

    def add(self, schema_name: str, params: Optional[Dict[str, Any]]=None,
            dependencies: Optional[List[str]]=None, strategy: Optional[str]=None) -> bool:
        if dependencies is None:
            dependencies = []
        if schema_name in self._schemas:
            raise KeyError(f"already registered schema: `{schema_name}`")
        schema = Schema(schema_name, self._schema_type, params, dependencies, strategy)
        self._schemas[schema_name] = schema
        return True

    def get(self, schema_name: str) -> Schema:
        if schema_name not in self._schemas:
            raise KeyError(f"unknown schema: `{schema_name}`")
        return self._schemas[schema_name]

    def contains(self, schema_name: str) -> bool:
        return schema_name in self._schemas

    def update(self, schema: Schema) -> bool:
        if not self.contains(schema.name):
            raise ValueError(f"unknown schema: `{schema.name}`")
        self._schemas[schema.name] = schema
        return True


class StrategyContainer:
    def __init__(self, schema_type: SchemaType):
        self._schema_type = schema_type
        self._strategies: Dict[str, SchemaStrategy] = {}

    @property
    def entry_count(self) -> int:
        return len(self._strategies)

    @property
    def is_empty(self) -> bool:
        return self.entry_count < 1

    def add(self, strategy_name: str, schema_type: SchemaType,
            body: Callable[[Any], Any], schema_link: Optional[str]=None) -> bool:
        if strategy_name in self._strategies:
            raise KeyError(f"already registered strategy: `{strategy_name}`")
        strategy = SchemaStrategy(strategy_name, schema_type, body, schema_link)
        self._strategies[strategy_name] = strategy
        return True

    def get(self, strategy_name: str) -> SchemaStrategy:
        if strategy_name not in self._strategies:
            raise KeyError(f"unknown strategy: `{strategy_name}`")
        return self._strategies[strategy_name]

    def contains(self, strategy_name: str) -> bool:
        return strategy_name in self._strategies

    def update(self, strategy: SchemaStrategy) -> bool:
        if not self.contains(strategy.name):
            raise ValueError(f"unknown strategy: `{strategy.name}`")
        self._strategies[strategy.name] = strategy
        return True