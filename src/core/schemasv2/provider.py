from typing import Dict, Any, Union

from core.schemas import SchemaType


class BuildProvider:
    def __init__(self, init_dict: Dict[str, Any]):
        self._collection = init_dict

        self._generate_get_methods()

    def _generate_get_methods(self):
        types = list(map(lambda key: key.split(".")[0], self._collection.keys()))
        for t in types:
            method_name = f"get_{t}"
            t = SchemaType.recognize(t)
            if not hasattr(self, method_name):
                def make_get(schema_type):
                    def get_method(self, schema_name: str):
                        return self.get(schema_type, schema_name)
                    return get_method
                self.__setattr__(method_name, make_get(t).__get__(self))

    def get(self, schema_type: Union[str, SchemaType], schema_name: str) -> Any:
        if isinstance(schema_type, SchemaType):
            schema_type = str(schema_type.value).lower()
        key = f"{schema_type}.{schema_name}"
        if key not in self._collection:
            raise KeyError(f"Unknown schema: `{key}`")
        return self._collection[key]