from ml_training_suite.registry import Registry, IncludeRegistry

class ML_Element(metaclass=IncludeRegistry):
    registry:Registry

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__()
        if register:
            cls.registry.register(cls, **kwargs)

    @classmethod
    def initialize(cls, obj:type|str=None, config:dict=None):
        if config is None:
            config = {}
        if obj is None:
            obj = cls
        return cls.registry.initialize(obj, config)
