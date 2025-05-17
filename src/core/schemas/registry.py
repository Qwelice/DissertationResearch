from typing import Dict, Any, Optional, Callable, List

from core.schemas import SchemaType
from core.schemas.containers import SchemaContainer


class Registry:
    def __init__(self, schema_types: List[SchemaType]):
        self._containers: Dict[SchemaType, SchemaContainer] = {}

        self._init_containers_(schema_types)

    def _init_containers_(self, schema_types: List[SchemaType]):
        for t in schema_types:
            self._containers[t] = SchemaContainer(schema_type=t)

    def _get_container_(self, schema_type: SchemaType) -> SchemaContainer:
        if schema_type not in self._containers.keys():
            raise ValueError(f"Unknown schema type: `{str(schema_type.value)}`")
        return self._containers[schema_type]

    def dependency(self, schema_name: str, schema_type: SchemaType) -> str:
        if schema_type not in self._containers:
            raise TypeError(f"Unknown schema type: `{schema_type.value}`")
        if schema_name not in self._containers[schema_type]:
            raise ValueError(f"Schema `{str(schema_type.value)}.{schema_name}` is not registered")
        return f"{str(schema_type.value)}.{schema_name}"

    def register_schema(self, schema_name: str, schema_type: SchemaType,
            params: Optional[Dict[str, Any]]=None, dependencies: Optional[List[str]]=None):
        def decorator(strategy: Callable[..., Any]):
            self._get_container_(schema_type).add(schema_name=schema_name,
                                                  schema_type=schema_type,
                                                  strategy=strategy,
                                                  params=params,
                                                  dependencies=dependencies)
            return strategy
        return decorator

    def build(self, schema_name: str, schema_type: SchemaType) -> Any:
        return self._get_container_(schema_type).build(schema_name, self)