from ml_training_suite.registry import Registry, IncludeRegistry

from typing import Union

class Config:
    def asDict(self):
        return self.__dict__

class ML_Element(metaclass=IncludeRegistry):
    registry:Registry

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__()
        if register:
            cls.registry.register(cls, **kwargs)

    @classmethod
    def initialize(cls, obj:Union[type,str]=None, config:Union[dict,Config]=None):
        if config is None:
            config = {}
        if isinstance(config, Config):
            config = config.asDict()
        if obj is None:
            obj = cls
        return cls.registry.initialize(obj, config)
