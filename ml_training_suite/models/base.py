from ml_training_suite.base import ML_Element, Config
from ml_training_suite.registry import Registry, IncludeRegistryABC

from typing import (
    Any,
    Tuple,
    Type,
    Union,
    Literal
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
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2

from pathlib import Path

class TorchModelSaveStructure:
    CLASS = "model_class"
    CONFIG = "model_config"
    STATE_DICT = "model_state_dict"

    def __init__(
            self,
            model_class,
            config,
            state_dict,) -> None:
        self.parts = {
            self.CLASS: model_class,
            self.CONFIG: config,
            self.STATE_DICT: state_dict}

    def __getitem__(self, key):
        return self.parts[key]

    @classmethod
    def compose_from_parts(
            cls,
            model_class:Union['Model',Type['Model']],
            config:'ModelConfig',
            state_dict:dict):
        if isinstance(model_class, Model):
            model_class = type(model_class)
        return cls(model_class, config, state_dict)
    
    @classmethod
    def compose_from_model(
            cls,
            model:'Model'):
        return cls.compose_from_parts(
            model,
            model.config,
            model.state_dict())

    def decompose_parts(
            self) -> Tuple[Type['Model'], 'ModelConfig', dict]:
        return self[self.CLASS], self[self.CONFIG], self[self.STATE_DICT]

class ModelConfig(Config):
    pass

class Model(nn.Module, ML_Element, register=False):
    registry = Registry()
    config:ModelConfig

    @classmethod
    def initialize(cls, obj:Union[type,str]=None, config:Union[dict,Config]=None):
        if (isinstance(obj, str) or isinstance(obj, Path)) and Path(obj).exists():
            return cls.load_model(obj)
        else:
            if config is None:
                config = {}
            if isinstance(config, Config):
                config = config.asDict()
            if obj is None:
                obj = cls
            return cls.registry.initialize(obj, config)

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
                           and len(list(enumerate(self)))==len(list(enumerate(decoder))))
        return autoencodable

    def _item_slicing_and_indexing(self, key):
        """
        How to divvy up the model.

        A definition of how the model should be indexed and sliced for purposes
        like iterative training over a growing portion of the model.

        Example:
            ::

            def _item_slicing_and_indexing(self):
                return torch.nn.Sequential(
                    self.model_preprocess,
                    self.model_body[key],
                    self.model_postprocess)
        """
        raise NotImplementedError((
            "_item_slicing_and_indexing must be overridden for {}"
            " to specify model indexing and slicing behavior.")
            .format(self))
    
    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key, int):
            return self._item_slicing_and_indexing(key)
        else:
            raise TypeError("Expected int or slice, but object {} is type {}."
                            .format(key, type(key)))
        
    # def __len__(self):
    #     return len(self[:])-len(self[:0])
    
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
    def load_onnx_model(load_path, cuda=True):
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

    @staticmethod
    def load_pytorch_model(load_path, cuda=True):
        model = torch.load(
            load_path,
            map_location=torch.device("cuda" if cuda else "cpu"))
        if isinstance(model, TorchModelSaveStructure):
            model_class, model_config, model_state_dict = model.decompose_parts()
            model = model_class(model_config)
            model.load_state_dict(model_state_dict)
        model.to(device=torch.device("cuda" if cuda else "cpu"))
        return model

    @staticmethod
    def load_model(load_path, cuda=False):
        load_path = Path(load_path)
        model = None
        if load_path.suffix in ['.onnx']:
            model = Model.load_onnx_model(load_path=load_path, cuda=cuda)
        elif load_path.suffix in ['.pt','.pth']:
            model = Model.load_pytorch_model(load_path=load_path, cuda=cuda)
        return model

    def save_pytorch_model(self, save_path:Path)->Path:
        save_file = save_path/"model.pt"
        torch.save(TorchModelSaveStructure.compose_from_model(self), save_file)
        return save_file

    def save_onnx_model(self, save_path:Path, data_input_sample:dict)->Path:
        save_file = save_path/"model.onnx"
        torch.onnx.export(
            self,
            tuple(data_input_sample.values()),
            input_names=list(data_input_sample.keys()),
            f = save_file,
            dynamic_axes={k: {0: "batch"} for k in data_input_sample.keys()}
        )
        return save_file

    def save_model(
            self,
            save_path:Path,
            save_as:Literal["torch","onnx"]="torch",
            data_input_sample=None,
            )->Path:
        save_file = None
        if save_as == "onnx":
            save_file = self.save_onnx_model(save_path, data_input_sample)
        elif save_as == "torch":
            save_file = self.save_pytorch_model(save_path)
        return save_file

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



class PolicyConfig(Config):
    pass

class Policy(TorchPolicyV2, ML_Element, metaclass=IncludeRegistryABC, register=False):
    registry = Registry()

    def __init_subclass__(cls, register=True, **kwargs):
        """Due to the following error: "WARNING policy.py:137 -- Can not figure out a
        durable policy name for <class 'my_chess.learner.policies.ppo.PPOPolicy'>.
        You are probably trying to checkpoint a custom policy. Raw policy class may cause
        problems when the checkpoint needs to be loaded in the future. To fix this, make
        sure you add your custom policy in rllib.algorithms.registry.POLICIES." We provide:"""
        import sys
        import ray.rllib.algorithms

        class_path = '/'.join(sys.modules[cls.__module__].__file__.split('/')[:-1])               
        if not class_path in ray.rllib.algorithms.__path__:
            ray.rllib.algorithms.__path__.append(class_path)

        from ray.rllib.algorithms.registry import POLICIES
        POLICIES[cls.__name__] = cls.__module__.split('.')[-1]

        return super().__init_subclass__(register, **kwargs)