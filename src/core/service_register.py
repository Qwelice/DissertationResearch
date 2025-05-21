from typing import Dict, Any


class ServiceRegistry:
    def __init__(self):
        self._services: Dict[str, Any] = {}

    def register(self, name: str, service: Any):
        self._services[name] = service

    def get(self, name: str) -> Any:
        return self._services[name]

    def contains(self, name: str) -> bool:
        return name in self._services