from typing import Callable, Any, Optional, Type

from core.schematic import SchemaType


class SchemaStrategy:
    def __init__(self, strategy_name: str, schema_type: SchemaType,
                 output_type: Type, body: Callable[[Any], Any]):
        self._strategy_name = strategy_name
        self._schema_type = schema_type
        self._output_type = output_type
        self._body = body

    @property
    def name(self) -> str:
        return self._strategy_name

    @property
    def schema_type(self) -> SchemaType:
        return self._schema_type

    @property
    def output_type(self) -> Type:
        return self._output_type

    def __call__(self, *args, **kwargs) -> Any:
        return self._body(*args, **kwargs)