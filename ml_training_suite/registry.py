from typing import (
    TypeVar,
    Union,
    List
)

T = TypeVar("T")
IterOptional = Union[T, List[T]]

class Registry:
    def __init__(self, *args, **kwargs):
        self._registry = {}
        for arg in args:
            self.register(arg)
        for k, arg in kwargs.items():
            self.register(arg, k)

    def register(self, cls, name=None):
        name = name if name else cls.__name__
        self._registry[name] = cls
        setattr(self, name, cls)
        return cls

    def unregister(self, cls):
        name = cls if isinstance(cls, str) else cls.__name__
        if name in self._registry:
            del self._registry[name]
            delattr(self, name)

    def __getitem__(self, key):
        return self._registry[key]

    def __iter__(self):
        return iter(self._registry.values())

    def __contains__(self, item):
        return item in self._registry

    def initialize(self, obj:Union[type, str], config:dict):
        if isinstance(obj, str):
            obj = getattr(self, obj)
        return obj(**config)

class IncludeRegistry(type):
    def __getattr__(cls, name):
        if name not in cls.__dict__:
            return getattr(cls.registry, name)
        else:
            return super().__getattr__(cls, name)