from typing import Dict, Optional, List, Any, Union, Type, Callable

from core.schematic.enums import SchemaType
from core.schematic import SchemaContainer, StrategyContainer, Schema, SchemaStrategy

CONTAINERS = Union[SchemaContainer, StrategyContainer]


class Registry:
    _SCHEMA='schema'
    _STRATEGY='strategy'

    def __init__(self):
        self._storage: Dict[SchemaType, Dict[str, CONTAINERS]] = {}

    @property
    def registered_types(self) -> List[SchemaType]:
        return self._storage.keys()

    @classmethod
    def _new_storage_item(cls, schema_type) -> dict[str, CONTAINERS]:
        return {
            cls._SCHEMA: SchemaContainer(schema_type),
            cls._STRATEGY: StrategyContainer(schema_type)
        }

    def _schemas(self, schema_type: SchemaType) -> SchemaContainer:
        if schema_type not in self._storage:
            raise TypeError(f"unknown schema type: `{schema_type}`")
        return self._storage[schema_type][self._SCHEMA]

    def _strategies(self, schema_type: SchemaType) -> StrategyContainer:
        if schema_type not in self._storage:
            raise TypeError(f"unknown schema type: `{schema_type}`")
        return self._storage[schema_type][self._STRATEGY]

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

    def register_strategy(self, strategy_name: str, schema_type: SchemaType,
                          body: Callable[[Any], Any], schema_link: Optional[str]=None) -> bool:
        if schema_type not in self._storage:
            raise TypeError(f"unknown schema type: `{schema_type}`")
        return self._storage[schema_type][self._STRATEGY].add(strategy_name=strategy_name,
                                                              schema_type=schema_type,
                                                              schema_link=schema_link,
                                                              body=body)

    def use(self, schema_name: str, schema_type: SchemaType, strategy_name, output_type: Type) -> bool:
        if schema_type not in self._storage:
            raise TypeError(f"unknown schema type: `{schema_type}`")
        if not self._schemas(schema_type).contains(schema_name):
            raise KeyError(f"unregistered schema: `{schema_name}`")
        if not self._strategies(schema_type).contains(strategy_name):
            raise KeyError(f"unregistered strategy: `{strategy_name}`")
        schema = self._schemas(schema_type).get(schema_name)
        schema.link_strategy(strategy_name, output_type)
        self._schemas(schema_type).update(schema)

    def get(self, name: str, schema_type: SchemaType, is_schema: bool=True) -> Union[Schema, SchemaStrategy]:
        if schema_type not in self._storage:
            raise TypeError(f"unknown schema type: `{schema_type}`")
        if is_schema:
            if not self._schemas(schema_type).contains(name):
                raise KeyError(f"unregistered schema: `{name}`")
            result = self._schemas(schema_type).get(name)
        else:
            if not self._strategies(schema_type).contains(name):
                raise KeyError(f"unregistered strategy: `{name}`")
            result = self._strategies(schema_type).get(name)

        return result

    def get_schema(self, schema_name: str, schema_type: SchemaType) -> Schema:
        return self.get(schema_name, schema_type, is_schema=True)

    def get_strategy(self, strategy_name: str, schema_type: SchemaType) -> SchemaStrategy:
        return self.get(strategy_name, schema_type, is_schema=False)

    def get_schemas(self, schema_type: SchemaType) -> List[Schema]:
        if schema_type not in self._storage:
            raise TypeError(f"unknown schema type: `{schema_type}`")
        return self._schemas(schema_type).getall()

    def get_strategies(self, schema_type) -> List[SchemaStrategy]:
        if schema_type not in self._storage:
            raise TypeError(f"unknown schema type: `{schema_type}`")
        return self._strategies(schema_type).getall()