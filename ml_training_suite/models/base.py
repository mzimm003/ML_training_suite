from ml_training_suite.base import ML_Element, Config
from ml_training_suite.registry import Registry

from typing import (
    Any,
)

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
from sklearn.decomposition import (
    PCA,
    FastICA
)

import torch
from torch import nn

import onnx
import onnxruntime

import numpy as np

import gymnasium as gym

from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class ModelConfig(Config):
    pass

class Model(nn.Module, ML_Element, register=False):
    registry = Registry()

    @staticmethod
    def __method_is_overridden(method:str, instance:'Model'):
        base_method = getattr(Model, method)
        instance_method = getattr(instance.__class__, method)
        return not (base_method is instance_method)

    def is_trainable_layer_wise(self:'Model'):
        return (
            Model.__method_is_overridden(
            Model._item_slicing_and_indexing.__name__,
            self)
            and self[:])

    def is_autoencodable(self:'Model'):
        autoencodable = Model.__method_is_overridden(
            Model.decoder.__name__,
            self)
        if self.is_trainable_layer_wise():
            decoder = self.decoder()
            autoencodable = (autoencodable
                           and decoder.is_trainable_layer_wise())
            autoencodable = (autoencodable
                           and len(self)==len(decoder))
        return autoencodable

    def _item_slicing_and_indexing(self, key):
        """
        How to divvy up the model.

        A definition of how the model should be indexed and sliced for purposes
        like iterative training over a growing portion of the model.

        Example:
            ::

            def _sliced_and_indexed_body(self):
                return torch.nn.Sequential(
                    self.model_preprocess,
                    self.model_body[key],
                    self.model_postprocess)
        """
        raise NotImplementedError((
            "_sliced_and_indexed_body must be overridden for {}"
            " to specify model indexing and slicing behavior.")
            .format(self))
    
    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key, int):
            return self._item_slicing_and_indexing(key)
        else:
            raise TypeError("Expected int or slice, but object {} is type {}."
                            .format(key, type(key)))
        
    def __len__(self):
        return len(self[:])-len(self[:0])
    
    def decoder(self) -> 'Model':
        """
        Mirror of model.

        To support autoencoder training a model must have a decoder version of
        itself. Decoder method must be implemented for any model intended to be
        trained by autoencoding.
        """
        raise NotImplementedError((
            "decoder must be overridden for {}"
            " to specify model's means for reconstructing input.")
            .format(self))

    @staticmethod
    def load_model(load_path, cuda=True):
        onnx.checker.check_model(onnx.load(load_path))
        class Mdl:
            type_to_np = {
                "tensor(double)":np.float64,
                "tensor(float)":np.float32,
                "tensor(float16)":np.float16
            }
            type_to_torch = {
                "tensor(double)":torch.float64,
                "tensor(float)":torch.float32,
                "tensor(float16)":torch.float16
            }
            def __init__(slf, load_path=load_path) -> None:
                slf.load_path = load_path
                slf.model = onnxruntime.InferenceSession(slf.load_path, providers = [
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider',
                    ] if cuda else ['CPUExecutionProvider'])
                slf.binding = slf.model.io_binding()

            def __getstate__(slf):
                return {'load_path': slf.load_path}

            def __setstate__(slf, values):
                slf.load_path = values['load_path']
                slf.model = onnxruntime.InferenceSession(load_path, providers = [
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider',
                    ] if cuda else ['CPUExecutionProvider'])
                slf.binding = slf.model.io_binding()

            def __call__(slf, *args: Any, **kwds:torch.Tensor) -> Any:
                # Assume positional args are fed for expected keywords
                if not kwds and args:
                    expected_kwds = [k.name for k in slf.model.get_inputs()]
                    for i, key in enumerate(expected_kwds):
                        kwds[key] = args[i]
                
                val_samp = None
                for arg in slf.model.get_inputs():
                    value = kwds[arg.name].to(dtype=Mdl.type_to_torch[arg.type]).contiguous()
                    val_samp = value
                    slf.binding.bind_input(
                        name=arg.name,
                        device_type='cuda' if cuda else 'cpu',
                        device_id=0,
                        element_type=Mdl.type_to_np[arg.type],
                        shape=value.shape,
                        buffer_ptr=value.data_ptr(),
                    )
                outs = []
                for op in slf.model.get_outputs():
                    out_shape = (val_samp.shape[0], *op.shape[1:])
                    out = torch.empty(
                        out_shape,
                        dtype=val_samp.dtype,
                        device=val_samp.device)
                    slf.binding.bind_output(
                        name=op.name,
                        device_type='cuda' if cuda else 'cpu',
                        device_id=0,
                        element_type=Mdl.type_to_np[op.type],
                        shape=out_shape,
                        buffer_ptr=out.data_ptr(),
                    )
                    outs.append(out)
                slf.model.run_with_iobinding(slf.binding)
                return outs
        return Mdl()

class ModelRLLIBConfig(ModelConfig):
    pass

class ModelRLLIB(TorchModelV2, Model, register=False):
    registry = Registry()

    def __init__(
        self,
        obs_space: gym.spaces.Space = None,
        action_space: gym.spaces.Space = None,
        num_outputs: int = None,
        model_config: ModelConfigDict = None,
        name: str = None):
        super().__init__(
            obs_space = obs_space,
            action_space = action_space,
            num_outputs = num_outputs,
            model_config = model_config,
            name = name
            )
        #super is not calling nn.Module init for unknown reasons
        nn.Module.__init__(self)

class FeatureReducer(ML_Element, register=False):
    registry = Registry(
        LDA = LinearDiscriminantAnalysis, # Reduces to fewer than number of classifications, only suitable for problems of multiple classes
        PCA = PCA,
        ICA = FastICA,
    )