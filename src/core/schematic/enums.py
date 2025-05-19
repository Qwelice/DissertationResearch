from enum import Enum


class SchemaType(Enum):
    MODULE='module'
    MODEL='model'
    DATASET='dataset'
    OPTIMIZER='optimizer'
    TRAINER='trainer'
    LOSSFN='lossfn'
    SCHEDULER='scheduler'
    METRIC='metric'
    LOOP='loop'
    TRANSFORM='transform'

    @classmethod
    def recognize(cls, value: str) -> 'SchemaType':
        for tp in SchemaType:
            if tp.value == value:
                return tp
        raise TypeError(f"no any schema type matching to value: `{value}`")