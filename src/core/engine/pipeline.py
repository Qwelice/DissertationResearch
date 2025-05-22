from core.configuration import PipelineConfig
from core.schematic import SchemaFactory
from core.service_register import ServiceRegistry
from core.services import BaseService


class Pipeline:
    def __init__(self, configuration: PipelineConfig):
        self._config = configuration
        self._factory = SchemaFactory(self._config._current_registry)
        self._services = ServiceRegistry()

    def use_service(self, svc_name: str, impl: BaseService):
        if self._services.contains(svc_name):
            raise KeyError(f"service `{svc_name}` is already registered")
        self._services.register(svc_name, impl)

    