from typing import Dict, Optional, Any, List

from core.schematic import SchemaType, Schema


class SchemaContainer:
    def __init__(self, schema_type: SchemaType):
        self._schema_type = schema_type
        self._schemas: Dict[str, Schema] = {}

    def add(self, schema_name: str, params: Optional[Dict[str, Any]]=None,
            dependencies: Optional[List[str]]=None, strategy: Optional[str]=None) -> bool:
        if dependencies is None:
            dependencies = []
        if schema_name in self._schemas:
            raise KeyError(f"already registered schema: `{schema_name}`")
        schema = Schema(schema_name, self._schema_type, params, dependencies, strategy)
        self._schemas[schema_name] = schema

    def get(self, schema_name: str) -> Schema:
        if schema_name not in self._schemas:
            raise KeyError(f"unknown schema: `{schema_name}`")
        return self._schemas[schema_name]