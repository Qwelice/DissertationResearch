from typing import Optional, Dict, Any, List

from core.schemas import Registry, SchemaType


class _Configuration:
    def __init__(self):
        types = self._get_supported_schema_types()

        self._registry = Registry(types)

    def _get_supported_schema_types(self):
        return [
            SchemaType.MODULE,
            SchemaType.DATASET,
            SchemaType.LOSSFN,
            SchemaType.LOOP,
            SchemaType.METRIC,
            SchemaType.REGIME,
            SchemaType.NONE
        ]

    def register_module(self, schema_name: str, params: Optional[Dict[str, Any]]=None,
                        dependencies: Optional[List[str]]=None):
        return self._registry.register_schema(schema_name=schema_name,
                                              schema_type=SchemaType.MODULE,
                                              params=params,
                                              dependencies=dependencies)

    def register_dataset(self, schema_name: str, params: Optional[Dict[str, Any]]=None,
                        dependencies: Optional[List[str]]=None):
        return self._registry.register_schema(schema_name=schema_name,
                                              schema_type=SchemaType.DATASET,
                                              params=params,
                                              dependencies=dependencies)

    def register_lossfn(self, schema_name: str, params: Optional[Dict[str, Any]] = None,
                        dependencies: Optional[List[str]] = None):
        return self._registry.register_schema(schema_name=schema_name,
                                              schema_type=SchemaType.LOSSFN,
                                              params=params,
                                              dependencies=dependencies)

    def register_loop(self, schema_name: str, params: Optional[Dict[str, Any]]=None,
                        dependencies: Optional[List[str]]=None):
        return self._registry.register_schema(schema_name=schema_name,
                                              schema_type=SchemaType.LOOP,
                                              params=params,
                                              dependencies=dependencies)

    def register_metric(self, schema_name: str, params: Optional[Dict[str, Any]]=None,
                        dependencies: Optional[List[str]]=None):
        return self._registry.register_schema(schema_name=schema_name,
                                              schema_type=SchemaType.METRIC,
                                              params=params,
                                              dependencies=dependencies)

    def register_regime(self, schema_name: str, params: Optional[Dict[str, Any]]=None,
                        dependencies: Optional[List[str]]=None):
        return self._registry.register_schema(schema_name=schema_name,
                                              schema_type=SchemaType.REGIME,
                                              params=params,
                                              dependencies=dependencies)

    def register_none(self, schema_name: str, params: Optional[Dict[str, Any]]=None,
                        dependencies: Optional[List[str]]=None):
        return self._registry.register_schema(schema_name=schema_name,
                                              schema_type=SchemaType.NONE,
                                              params=params,
                                              dependencies=dependencies)

    def build_module(self, schema_name: str):
        return self._registry.build(schema_name=schema_name,
                                    schema_type=SchemaType.MODULE)

    def build_dataset(self, schema_name: str):
        return self._registry.build(schema_name=schema_name,
                                    schema_type=SchemaType.DATASET)

    def build_lossfn(self, schema_name: str):
        return self._registry.build(schema_name=schema_name,
                                    schema_type=SchemaType.LOSSFN)

    def build_loop(self, schema_name: str):
        return self._registry.build(schema_name=schema_name,
                                    schema_type=SchemaType.LOOP)

    def build_metric(self, schema_name: str):
        return self._registry.build(schema_name=schema_name,
                                    schema_type=SchemaType.METRIC)

    def build_regime(self, schema_name: str):
        return self._registry.build(schema_name=schema_name,
                                    schema_type=SchemaType.REGIME)

Configuration = _Configuration()