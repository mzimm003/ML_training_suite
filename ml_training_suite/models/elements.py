from ml_training_suite.base import ML_Element
from ml_training_suite.registry import Registry

from torch import nn

class ActivationReg(ML_Element, register=False):
    registry = Registry(
        relu = nn.ReLU,
        sig = nn.Sigmoid,
    )