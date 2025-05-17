import inspect
from typing import Dict, List, Callable, Any, Optional

from core.schemas import Schema, SchemaType


class SchemaContainer:
    def __init__(self, schema_type: SchemaType=SchemaType.NONE):
        self._schemas: Dict[str, Schema] = dict()
        self._schema_type = schema_type

    @property
    def is_virtual(self) -> bool:
        return self._schema_type == SchemaType.NONE

    @property
    def schema_type(self) -> SchemaType:
        return self._schema_type

    def add(self, schema_name: str, schema_type: SchemaType,
            strategy: Callable[..., Any], params: Optional[Dict[str, Any]]=None,
            dependencies: Optional[List[str]]=None) -> None:
        if self.is_virtual:
            raise TypeError("Virtual container cannot store any schema")
        if self._schema_type != schema_type:
            raise TypeError(f"{str(self._schema_type.value).capitalize()} "
                            f"container cannot store {str(schema_type.value)} typed schema")
        if schema_name in self._schemas:
            raise ValueError(f"Schema `{str(self._schema_type.value)}.{schema_name}` is registered already")
        if dependencies is None:
            dependencies = []
        if not isinstance(strategy, Callable):
            raise TypeError(f"Schema `{str(self._schema_type.value)}.{schema_name}`: schema strategy must be callable")
        strategy_sig = inspect.signature(strategy)
        contains_params = "params" in strategy_sig.parameters
        if len(strategy_sig.parameters) - contains_params != len(dependencies):
            raise ValueError(f"Schema `{str(self._schema_type.value)}.{schema_name}`: "
                             f"expected {len(dependencies)} dependencies "
                             f"but got {len(strategy_sig)}")
        self._schemas[schema_name] = Schema(schema_name, schema_type, params, dependencies, strategy)

    def get(self, schema_name: str) -> Schema:
        if schema_name not in self._schemas.keys():
            raise KeyError(f"Schema `{str(self._schema_type.value)}.{schema_name}` is not found")
        return self._schemas[schema_name]

    def build(self, schema_name: str, registry: "Registry") -> Any:
        schema = self.get(schema_name)
        deps = []
        for dep in schema.dependencies:
            dep_parts = dep.split(".")
            if len(dep_parts) != 2:
                raise KeyError(f"Unknown dependency name: `{dep}` "
                               f"expected 2 parts by dot "
                               f"but got {len(dep_parts)}")
            dep_schema_type = SchemaType.recognize(dep_parts[0])
            dep_schema_name = dep_parts[1]
            dep_impl = registry.build(dep_schema_name, dep_schema_type)
            deps.append(dep_impl)
        schema_impl = schema.strategy(*deps, schema.params) if schema.params else schema.strategy(*deps)
        return schema_impl