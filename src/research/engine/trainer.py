class Trainer:
    def __init__(self, config):
        self._config = config
        self._data_cfg = config.data
        # self._model_cfg = config.model

    def _prepare_data_(self):
        ...