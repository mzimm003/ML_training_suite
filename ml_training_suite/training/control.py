from ml_training_suite.base import ML_Element
from ml_training_suite.registry import Registry

from torch.optim import (
    Optimizer as OptimizerTorch,
    SGD,
    Adam,
)
from torch.optim.lr_scheduler import (
    LRScheduler as LRSchedulerTorch,
    CyclicLR
)
from torch import nn

from typing import (
    Literal,
)

class Optimizer(OptimizerTorch, ML_Element, register=False):
    registry = Registry(
        SGD = SGD,
        adam = Adam,
    )

    @classmethod
    def initialize(cls, obj, parameters, kwargs):
        kwargs['params'] = parameters
        return super().initialize(obj, kwargs)

class LRSchedulerReg:
    def __init__(
            self,
            batch_based_registry=None,
            epoch_based_registry=None,):
        self._batch_based_registry = (
            Registry()
            if batch_based_registry is None
            else batch_based_registry)
        self._epoch_based_registry = (
            Registry()
            if epoch_based_registry is None
            else epoch_based_registry)
        self._registries = {
            'epoch':self._epoch_based_registry,
            'batch':self._batch_based_registry,
            }

    def register(self, cls, *, basis:Literal['epoch','batch'], name=None):
        return self._registries[basis].register(cls, name=name)

    def unregister(self, cls, basis=None):
        if not basis is None:
            self._registries[basis].unregister(cls=cls)
        else:
            for reg in self._registries.values():
                reg.unregister(cls=cls)

    def __getitem__(self, key):
        if key in self._epoch_based_registry:
            if key in self._batch_based_registry:
                raise RuntimeError("{} exists as Epoch and Batch based. Keys should be unique between bases.".format(key))
            return self._epoch_based_registry[key]
        else:
            return self._batch_based_registry[key]

    def __getattr__(self, name):
        if name in self._epoch_based_registry:
            if name in self._batch_based_registry:
                raise RuntimeError("{} exists as Epoch and Batch based. Keys should be unique between bases.".format(name))
            return self._epoch_based_registry[name]
        else:
            return self._batch_based_registry[name]

    def __iter__(self):
        return iter([
            *iter(self._epoch_based_registry),
            *iter(self._batch_based_registry),
        ])

    def __contains__(self, item):
        return (
            item in self._epoch_based_registry or
            item in self._batch_based_registry
        )   

    def initialize(self, obj:type|str, config:dict):
        if isinstance(obj, str):
            obj = getattr(self, obj)
        return obj(**config)
    
    def is_epoch_based(self, key):
        return key in self._epoch_based_registry
    
    def is_batch_based(self, key):
        return key in self._batch_based_registry

class LRScheduler(LRSchedulerTorch, ML_Element, register=False):
    registry:LRSchedulerReg = LRSchedulerReg(
        batch_based_registry=Registry(
            CyclicLR,
        )
    )

    @classmethod
    def initialize(cls, obj, optimizer, kwargs):
        kwargs['optimizer'] = optimizer
        return super().initialize(obj, kwargs)
    
    @classmethod
    def is_epoch_based(cls, key):
        return cls.registry.is_epoch_based(key)

    @classmethod
    def is_batch_based(cls, key):
        return cls.registry.is_batch_based(key)

class Criterion(ML_Element, register=False):
    registry = Registry(
        MSE = nn.MSELoss,
        cross_entropy = nn.CrossEntropyLoss,
    )