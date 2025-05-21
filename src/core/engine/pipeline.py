from core.configuration import PipelineConfig
from core.schematic import SchemaFactory
from core.service_register import ServiceRegistry


class Pipeline:
    def __init__(self, configuration: PipelineConfig):
        self._config = configuration
        self._factory = SchemaFactory(self._config._current_registry)
        self._services = ServiceRegistry()

