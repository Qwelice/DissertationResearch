from enum import Enum


class SchemaType(Enum):
    MODULE = "module"
    DATASET = "dataset"
    LOSSFN = "lossfn"
    METRIC = "metric"
    REGIME = "regime"
    LOOP = "loop"
    NONE = "none"

    @classmethod
    def recognize(cls, value: str) -> 'SchemaType':
        if value == str(SchemaType.MODULE.value):
            return SchemaType.MODULE
        elif value == str(SchemaType.DATASET.value):
            return SchemaType.DATASET
        elif value == str(SchemaType.LOSSFN.value):
            return SchemaType.LOSSFN
        elif value == str(SchemaType.METRIC.value):
            return SchemaType.METRIC
        elif value == str(SchemaType.REGIME.value):
            return SchemaType.REGIME
        elif value == str(SchemaType.LOOP.value):
            return SchemaType.LOOP
        else:
            return SchemaType.NONE