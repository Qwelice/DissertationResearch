from typing import Optional, Dict

from core.schematic.registry import Registry


class PipelineConfig:
    def __init__(self, project_name: Optional[str]=None):
        if project_name is None:
            project_name = 'default'
        project_name = project_name.strip().lower()
        self._current_project = project_name
        self._registries: Dict[str, Registry] = {
            project_name: Registry()
        }

    @property
    def current_project(self) -> str:
        return self._current_project

    def register_project(self, project_name: str) -> bool:
        project_name = project_name.strip().lower()
        if project_name in self._registries:
            raise KeyError(f"project already registered: `{project_name}`")
        self._registries[project_name] = Registry()
        return True

    def set_current_project(self, project_name: str) -> bool:
        project_name = project_name.strip().lower()
        if project_name not in self._registries:
            raise KeyError(f"unknown project: `{project_name}`")
        self._current_project = project_name
        return True

    def register_and_set_current_project(self, project_name: str) -> bool:
        project_name = project_name.strip().lower()
        preg = self.register_project(project_name)
        pset = self.set_current_project(project_name)
        return preg and pset

