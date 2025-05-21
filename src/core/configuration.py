from typing import Optional, Dict, Any, List, Type, Callable

from core.schematic.enums import SchemaType
from core.schematic.registry import Registry
from core.utils import PathLike, ConfigObject


class PipelineConfig:
    def __init__(self,
                 project_name: Optional[str]=None,
                 outputs_dir: Optional[PathLike]=None,
                 logs_dir: Optional[PathLike]=None,
                 ckpt_dir: Optional[PathLike]=None,
                 config: Optional[ConfigObject]=None):
        if project_name is None:
            project_name = 'default'
        project_name = project_name.strip().lower()
        self._current_project = project_name
        self._registries: Dict[str, Registry] = {
            project_name: Registry()
        }
        self._outputs_dir = outputs_dir
        self._logs_dir = logs_dir
        self._ckpt_dir = ckpt_dir
        self._config =config

    @property
    def outputs_dir (self) -> Optional[PathLike]:
        return self._outputs_dir

    @property
    def logs_dir (self) -> Optional[PathLike]:
        return self._logs_dir

    @property
    def ckpt_dir (self) -> Optional[PathLike]:
        return self._ckpt_dir

    @property
    def config (self) -> Optional[ConfigObject]:
        return self._config

    @property
    def _current_registry(self) -> Registry:
        return self._registries[self.current_project]

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

    def register_schema (self, schema_name: str, schema_type: SchemaType, params: Optional[Dict[str, Any]] = None,
                        dependencies: Optional[List[str]] = None, strategy: Optional[str] = None):
        self._current_registry.register_schema(schema_name, schema_type, params, dependencies, strategy)

    def register_strategy (self, strategy_name: str, schema_type: SchemaType,
                           body: Callable[[Any], Any], schema_link: Optional[str]=None):
        self._current_registry.register_strategy(strategy_name, schema_type, body, schema_link)

    def link_strategy (self, schema_name: str, schema_type: SchemaType, strategy_name: str, output_type: Type) -> bool:
        return self._current_registry.use(schema_name, schema_type, strategy_name, output_type)