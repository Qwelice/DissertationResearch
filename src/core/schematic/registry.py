from typing import Dict, Optional, List, Any, Union, Type

from core.schematic.enums import SchemaType
from core.schematic import SchemaContainer, StrategyContainer


CONTAINERS = Union[SchemaContainer, StrategyContainer]


class Registry:
    _SCHEMA='schema'
    _STRATEGY='strategy'

    def __init__(self):
        self._storage: Dict[SchemaType, Dict[str, CONTAINERS]] = {}

    @property
    def _schemas(self) -> SchemaContainer:
        return self._storage[self._SCHEMA]

    @property
    def _strategies (self) -> StrategyContainer:
        return self._storage[self._STRATEGY]

    @classmethod
    def _new_storage_item(cls, schema_type) -> dict[str, CONTAINERS]:
        return {
            cls._SCHEMA: SchemaContainer(schema_type),
            cls._STRATEGY: StrategyContainer(schema_type)
        }

    def register_type(self, schema_type: SchemaType) -> bool:
        if schema_type in self._storage:
            containers = self._storage[schema_type]
            if not containers[self._SCHEMA].is_empty or not containers[self._STRATEGY].is_empty:
                raise RuntimeError(f"{schema_type} containers already exist and they are not empty")
        containers = Registry._new_storage_item(schema_type)
        self._storage[schema_type] = containers

        return True

    def register_schema(self, schema_name: str, schema_type: SchemaType, params: Optional[Dict[str, Any]] = None,
                        dependencies: Optional[List[str]] = None, strategy: Optional[str] = None) -> bool:
        if schema_type not in self._storage:
            raise TypeError(f"unknown schema type: `{schema_type}`")
        return self._storage[schema_type][self._SCHEMA].add(schema_name, params, dependencies, strategy)

    def register_strategy(self, strategy_name: str, schema_type: SchemaType, schema_link: Optional[str]=None):
        def decorator(fn):
            if schema_type not in self._storage:
                raise TypeError(f"unknown schema type: `{schema_type}`")
            return self._storage[schema_type][self._STRATEGY].add(strategy_name=strategy_name,
                                                                  schema_type=schema_type,
                                                                  schema_link=schema_link,
                                                                  body=fn)
        return decorator

    def use(self, schema_name: str, schema_type: SchemaType, strategy_name, output_type: Type) -> bool:
        if schema_type not in self._storage:
            raise TypeError(f"unknown schema type: `{schema_type}`")
        if not self._schemas.contains(schema_name):
            raise KeyError(f"unregistered schema: `{schema_name}`")
        if not self._strategies.contains(strategy_name):
            raise KeyError(f"unregistered strategy: `{strategy_name}`")
        schema = self._schemas.get(schema_name)
        schema.link_strategy(strategy_name, output_type)
        self._schemas.update(schema)