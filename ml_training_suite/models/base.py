from ml_training_suite.base import ML_Element
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

class Model(nn.Module, ML_Element, register=False):
    registry = Registry()

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

class FeatureReducer(ML_Element, register=False):
    registry = Registry(
        LDA = LinearDiscriminantAnalysis, # Reduces to fewer than number of classifications, only suitable for problems of multiple classes
        PCA = PCA,
        ICA = FastICA,
    )