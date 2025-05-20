from typing import Callable, Any, Optional

from core.schematic import SchemaType


class SchemaStrategy:
    def __init__(self, strategy_name: str, schema_type: SchemaType,
                 body: Callable[[Any], Any], schema_link: Optional[str]=None):
        self._strategy_name = strategy_name
        self._schema_type = schema_type
        self._body = body
        self._schema_link = schema_link

    @property
    def name(self) -> str:
        return self._strategy_name

    @property
    def schema_type(self) -> SchemaType:
        return self._schema_type

    @property
    def is_linked(self) -> bool:
        return self._schema_link is not None

    @property
    def schema_link(self) -> Optional[str]:
        return self._schema_link

    def link_with(self, schema_link: str):
        self._schema_link = schema_link

    def __call__(self, *args, **kwargs) -> Any:
        return self._body(*args, **kwargs)