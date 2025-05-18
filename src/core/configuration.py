from typing import Optional, Dict, Any, List

from core.schemas import Registry, SchemaType, BuildProvider


class _Configuration:
    def __init__(self):
        types = self._get_supported_schema_types()

        self._is_built: bool = False
        self._provider: Optional[BuildProvider] = None
        self._registry: Registry = Registry(types)
        self._generate_register_methods(types)
        self._generate_dependency_methods(types)

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

    def _generate_register_methods(self, types: List[SchemaType]):
        for t in types:
            method_name = f"register_{str(t.value).lower()}"
            if not hasattr(self, method_name):
                def make_register(schema_type):
                    def register_method(self, schema_name: str, params: Optional[Dict[str, Any]] = None,
                                        dependencies: Optional[List[str]] = None):
                        return self._registry.register_schema(
                            schema_name=schema_name,
                            schema_type=schema_type,
                            params=params,
                            dependencies=dependencies
                        )

                    return register_method
                self.__setattr__(method_name, make_register(t).__get__(self))

    def _generate_dependency_methods(self, types: List[SchemaType]):
        for t in types:
            depmethod_name = f"dependency_{str(t.value).lower()}"
            regmethod_name = f"registered_{str(t.value).lower()}"
            if not hasattr(self, depmethod_name) or not hasattr(self, regmethod_name):
                def make_dependency(schema_type: SchemaType):
                    def dependency_method(self, schema_name: str):
                        return self._registry.dependency(schema_name=schema_name, schema_type=schema_type)
                    return dependency_method
                if not hasattr(self, depmethod_name):
                    self.__setattr__(depmethod_name, make_dependency(t).__get__(self))
                if not hasattr(self, regmethod_name):
                    self.__setattr__(regmethod_name, make_dependency(t).__get__(self))

    @property
    def is_built(self) -> bool:
        return self._is_built

    def build(self, actualize: Optional[bool]=None) -> BuildProvider:
        if actualize is None:
            actualize = False

        if self.is_built and not actualize:
            if self._provider is None:
                raise RuntimeError("Build provider is not initialized")
            return self._provider

        self._is_built = True
        from tqdm import tqdm
        containers = tqdm(self._registry._containers.items(), "schemas building", leave=True)
        for schema_type, container in containers:
            containers.set_postfix({ "container": str(schema_type.value).lower() })
            if container.is_virtual:
                continue
            for schema_name in container._schemas:
                self._registry.build(schema_name, schema_type)
        self._provider = BuildProvider(self._registry._cached)
        return self._provider

    def rebuild(self) -> BuildProvider:
        self._is_built = False
        return self.build(actualize=True)


Configuration = _Configuration()