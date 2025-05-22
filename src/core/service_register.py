from typing import Dict, Any

from core.services import BaseService


class ServiceRegistry:
    def __init__(self):
        self._services: Dict[str, BaseService] = {}

    def register(self, name: str, service: BaseService):
        self._services[name] = service

    def get(self, name: str) -> BaseService:
        return self._services[name]

    def contains(self, name: str) -> bool:
        return name in self._services