from typing import Any, Dict, Optional

from core.schematic import Registry, Schema, SchemaStrategy
from core.schematic.enums import SchemaType


class SchemaFactory:
    def __init__(self, registry: Registry):
        self._registry = registry
        self._cache: Dict[str, Any] = {}

    def _create_cache_key(self, schema_type: SchemaType, schema_name) -> str:
        return f"{schema_type.value}.{schema_name}"

    def build(self, schema_name: str, schema_type: SchemaType, seen: Optional[set]=None) -> Any:
        schema: Schema = self._registry.get_schema(schema_name, schema_type)
        strategy_name = schema.strategy
        if not strategy_name:
            raise ValueError(f"schema `{schema_name}` has no strategy linked")

        if not schema.output_type:
            raise ValueError(f"strategy `{strategy_name}`: output type is not identified")

        seen = seen or set()
        key = self._create_cache_key(schema_type, schema_name)
        if key in self._cache:
            return self._cache[key]
        if key in seen:
            raise RuntimeError(f"cyclic dependency detected at: {key}")
        seen.add(key)

        strategy: SchemaStrategy = self._registry.get_strategy(strategy_name, schema_type)

        dependencies = [self.build(dep, schema_type, seen=seen.copy()) for dep in (schema.dependencies or [])]

        output = strategy(*dependencies, **(schema.params or {}))

        if not isinstance(output, schema.output_type):
            raise TypeError(f"invalid strategy output type: expected {schema.output_type.__name__} "
                            f"but got {type(output).__name__}")

        self._cache[key] = output

        return output

    def build_all(self) -> Dict[str, Any]:
        for schema_type in self._registry.registered_types:
            for schema in self._registry.get_schemas(schema_type):
                self.build(schema_name=schema.name, schema_type=schema_type)
        return self._cache

    def clear_cache(self) -> None:
        self._cache.clear()

    def rebuild_all(self) -> Dict[str, Any]:
        self.clear_cache()
        return self.build_all()