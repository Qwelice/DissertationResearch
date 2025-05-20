from typing import Dict, Optional, Any, List, Callable

from core.schematic import SchemaType, Schema, SchemaStrategy


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