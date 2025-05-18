from core.schemas import Registry


class ConfigurationBuilder:
    def __init__(self, reg: Registry):
        self._registry = reg
