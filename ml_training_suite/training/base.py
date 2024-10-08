from ml_training_suite.base import ML_Element
from ml_training_suite.registry import Registry, IterOptional
from ml_training_suite.datasets import (
    Dataset,
    DataHandlerGenerator,
    Subset,
    train_test_split,
)
from ml_training_suite.callbacks import Callback
from ml_training_suite.models import Model

from pathlib import Path
import math
import json

from typing import (
    Type,
    Callable,
    Union,
    Any,
    Iterable,
    Literal,
    List
    )
from typing_extensions import override
from dataclasses import dataclass
import warnings

from ray import tune

import torch
import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import (
    Optimizer as OptimizerTorch,
    SGD,
    Adam,
)
from torch.optim.lr_scheduler import (
    LRScheduler as LRSchedulerTorch,
    CyclicLR
)

from sklearn.model_selection import StratifiedKFold
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn

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

    def initialize(self, obj:Union[type,str], config:dict):
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

def ray_trainable_wrap(
        script:Type["TrainingScript"] = None,
        num_cpus:int=1,
        num_gpus:float=0.):
    class TrainableWrapper(tune.Trainable):
        @override
        def setup(self, config:dict):
            self.script = script(**config)
            self.script.setup()
        @override
        def step(self):
            return self.script.run()
        @override
        def save_checkpoint(self, checkpoint_dir: str) -> Union[dict, None]:
            for i, (mod, data_samp) in enumerate(self.script.get_models_for_onnx_save()):
                self.script.save_model(mod, data_samp, i, save_dir=checkpoint_dir)
    return tune.with_resources(TrainableWrapper, resources={"CPU":num_cpus, "GPU":num_gpus})

class Trainer(ML_Element, register=False):
    registry = Registry()
    def __init__(
            self,
            model:nn.Module,
            optimizer:Optimizer,
            criterion:torch.nn.modules.loss._Loss,
            lr_scheduler:LRScheduler=None,
            pipeline=None):
        self.model = model
        param_ref = next(iter(model.parameters()))
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.device = param_ref.device
        self.dh_gen:DataHandlerGenerator = self.map_data_handler(pipeline)
    
    def map_data_handler(self, pipeline):
        return DataHandlerGenerator(pipeline=pipeline if pipeline else [])
    
    def generate_data_handler(self, data):
        return self.dh_gen.get_data_handler(data)
    
    def step_lr_scheduler(self, endOfEpoch=False):
        if self.lr_scheduler:
            if (endOfEpoch and
                LRScheduler.is_epoch_based(self.lr_scheduler.__class__.__name__) or
                (not endOfEpoch and
                LRScheduler.is_batch_based(self.lr_scheduler.__class__.__name__))
                ):
                self.lr_scheduler.step()

    def infer_itr(
            self,
            data
            ):
        data_handler = self.generate_data_handler(data)
        data_handler.process_data(self.model)

        if self.model.training:
            self.optimizer.zero_grad()
        
        with torch.autocast(device_type=self.device.type):
            data_handler.set_model_output(self.model(**data_handler.get_inputs()))
            
            data_handler.set_loss(self.criterion)

        if self.model.training:
            data_handler.loss.backward()
            self.optimizer.step()
            self.step_lr_scheduler()
        if not self.lr_scheduler is None:
            data_handler.set_last_lr(self.lr_scheduler.get_last_lr()[0])
        else:
            data_handler.set_last_lr(next(iter(self.optimizer.param_groups))['lr'])
        return data_handler

@dataclass
class TrainSplits:
    train_data:DataLoader
    val_data:DataLoader
    trainers:List['Trainer']

class TrainingManager:
    def __init__(
        self,
        data:Dataset,
        dl_kwargs:dict,
        num_splits:int,
        balance_training_set:bool,
        shuffle:bool,
        trainer_class:Type['Trainer'],
        pipelines:IterOptional[List[tuple[str, Callable]]] = None,
        models:IterOptional[Union[str, Type[Model]]] = None,
        models_kwargs:IterOptional[dict[str, Any]] = None,
        optimizers:IterOptional[Type[Optimizer]]= None,
        optimizers_kwargs:IterOptional[dict[str, Any]] = None,
        lr_schedulers:IterOptional[Type[LRScheduler]]= None,
        lr_schedulers_kwargs:IterOptional[dict[str, Any]] = None,
        criterion:Union[Criterion, Type[torch.nn.modules.loss._Loss]]= None,
        criterion_kwargs:dict[str, Any] = None,
        ):
        self.num_splits = num_splits
        self.balance_training_set = balance_training_set
        self.shuffle = shuffle
        self.data = data
        self.dl_kwargs = dl_kwargs
        self.trainer_class = trainer_class
        self.device = (torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        print("Expected device:{}".format(self.device))
        
        self.pipelines = pipelines
        self.models:IterOptional[nn.Module] = models
        if not isinstance(self.models, list):
            self.models = [self.models]
        self.models_kwargs = models_kwargs if models_kwargs else [{}]*len(self.models)
        if not isinstance(self.models_kwargs, list):
            self.models_kwargs = [self.models_kwargs]*len(self.models)
        self.optimizers = optimizers
        if not isinstance(self.optimizers, list):
            self.optimizers = [self.optimizers]*len(self.models)
        self.optimizers_kwargs = optimizers_kwargs if optimizers_kwargs else [{}]*len(self.models)
        if not isinstance(self.optimizers_kwargs, list):
            self.optimizers_kwargs = [self.optimizers_kwargs]*len(self.models)
        self.lr_schedulers = lr_schedulers
        if not isinstance(self.lr_schedulers, list):
            self.lr_schedulers = [self.lr_schedulers]*len(self.models)
        self.lr_schedulers_kwargs = lr_schedulers_kwargs if lr_schedulers_kwargs else [{}]*len(self.models)
        if not isinstance(self.lr_schedulers_kwargs, list):
            self.lr_schedulers_kwargs = [self.lr_schedulers_kwargs]*len(self.models)
        self.criterion = Criterion.initialize(
            criterion, criterion_kwargs)
        self.criterion.to(device=self.device)
        self.splits:dict[int, TrainSplits] = {}
        self.create_splits()
        assert len(self) == 1
        for split in self:
            for trainer_x in split.trainers:
                model_match_count = 0
                optimizer_match_count = 0
                for trainer_y in split.trainers:
                    if trainer_x.model == trainer_y.model:
                        model_match_count += 1
                    if trainer_x.optimizer == trainer_y.optimizer:
                        optimizer_match_count += 1
                assert model_match_count == 1
                assert optimizer_match_count == 1
    
    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError
        return self.splits[index]

    def create_splits(self):
        for i, (training_data, validation_data) in enumerate(self.create_dataloaders()):
            self.splits[i] = TrainSplits(
                train_data=training_data,
                val_data=validation_data,
                trainers=self.create_trainers())

    def create_dataloaders(self):
        split_idxs = None
        if self.num_splits == 1:
            split_idxs = train_test_split(
                self.data,
                train_size=0.8,
                test_size=0.2,
                shuffle=self.shuffle,
                stratify=[self.data.labels]
                )
        else:
            split_idxs = StratifiedKFold(
                n_splits=self.num_splits,
                shuffle=self.shuffle,
                ).split(self.data, self.data.labels)
            
        d_ls = []
        for train_fold, val_fold in split_idxs:
            print("Start dataloaders.")
            train_dl_kwargs = self.dl_kwargs.copy()
            train_data = Subset(self.data, train_fold)
            val_data = Subset(self.data, val_fold)
            if self.balance_training_set:
                unq_lbls = train_data.getLabels().unique()
                lbl_masks = train_data.getLabels()==unq_lbls[:,None]
                not_lbl_counts = (train_data.getLabels()!=unq_lbls[:,None]).sum(-1)[:,None]
                weights = (lbl_masks*not_lbl_counts).sum(0)
                train_dl_kwargs["sampler"] = (
                    torch.utils.data.WeightedRandomSampler(weights, len(train_data)))
                train_dl_kwargs["shuffle"] = False
            d_ls.append((DataLoader(train_data,**train_dl_kwargs),
                   DataLoader(val_data,**self.dl_kwargs)))
        return d_ls
        
    def create_trainers(self):
        ts = []
        for j, m in enumerate(self.models):
            model = Model.initialize(m, self.models_kwargs[j]).to(device=self.device)
            opt = Optimizer.initialize(
                    self.optimizers[j], model.parameters(), self.optimizers_kwargs[j])
            lr_sch = self.lr_schedulers[j]
            if lr_sch:
                lr_sch = LRScheduler.initialize(
                    lr_sch, opt, self.lr_schedulers_kwargs[j])
            ts.append(self.trainer_class(
                model=model,
                optimizer=opt,
                lr_scheduler=lr_sch,
                criterion=self.criterion,
                pipeline=self.pipelines[j]))
        return ts

    def get_models_for_onnx_save(self, dtype=None)-> tuple:
        mods = []
        for i, split in self.splits.items():
            data_sample = next(iter(split.train_data))
            for trainer in split.trainers:
                d_h = trainer.generate_data_handler(data_sample)
                d_h.process_data(trainer.model, dtype=dtype, trunc_batch=8)
                mods.append((trainer.model.eval().to(dtype=dtype), d_h.get_inputs()))
        return mods
    
    def step_lr_schedulers(self):
        for train_elements in self:
            for trainer in train_elements.trainers:
                trainer.step_lr_scheduler(endOfEpoch=True)

class TrainingScript(ML_Element, register=False):
    registry = Registry()
    data: Type[Dataset]
    save_path: Union[str, Path]
    training_manager: 'TrainingManager'
    callback:Callback

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = (torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def __isclose(input:torch.Tensor, other:torch.Tensor, equal_nan=False):
        """
        Based on the proposal found here: https://github.com/pytorch/pytorch/issues/41651,
        the need to measure whether a torch model and onnx model behave effectively
        the same without necessarily casting to greater floating point precisions,
        and the description of floating point values here: https://stackoverflow.com/questions/872544/what-range-of-numbers-can-be-represented-in-a-16-32-and-64-bit-ieee-754-syste,
        this method provides a measure of closeness dynamically based on dtype.
        """
        assert input.dtype == other.dtype
        E = int(math.log2(other.abs().max()))
        epsilon_adj = {
            torch.float16:-10,
            torch.float32:-23,
            torch.float64:-52,
        }
        epsilon = 2**(E-epsilon_adj[other.dtype])

        return torch.isclose(input=input,
                             other=other,
                             rtol=epsilon*20,
                             atol=epsilon*5,
                             equal_nan=equal_nan)

    def save_model(self, model, data_input_sample, suffix="", save_dir=None, validate_model=True):
        validation_output = None
        if validate_model:
            if isinstance(model, nn.Module):
                validation_output = model(**data_input_sample)
            else:
                validation_output = model.transform(data_input_sample)
                validation_output = torch.from_numpy(validation_output).float()
        if isinstance(validation_output, Iterable):
            validation_output = validation_output[0]

        onx = None
        save_path = Path(save_dir) if save_dir else self.save_path
        save_file = save_path / "{}{}/model.onnx".format(self.get_model_name(model), suffix)
        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)

        if isinstance(model, nn.Module):
            torch.onnx.export(
                model,
                tuple(data_input_sample.values()),
                input_names=list(data_input_sample.keys()),
                f = save_file,
                dynamic_axes={k: {0: "batch"} for k in data_input_sample.keys()}
            )
        else:
            init_types = [("fet", FloatTensorType([None, data_input_sample.shape[-1]]))]
            onx = convert_sklearn(model, initial_types=init_types)
            with open(save_file, "wb") as f:
                f.write(onx.SerializeToString())
        if validate_model:
            res_mod = None
            res_output = None
            if isinstance(model, nn.Module):
                res_mod = Model.load_model(save_file)
                res_output, *_ = res_mod(**data_input_sample)
            else:
                res_mod = Model.load_model(save_file, cuda=False)
                res_output, *_ = res_mod(data_input_sample)
            if not (validation_output.max(-1).indices==res_output.max(-1).indices).all():
                warnings.warn(
                    "Output decisions of training model and saved model do not match.",
                    UserWarning)
            if not self.__isclose(validation_output, res_output).all():
                warnings.warn(
                    "Output logits of training model and saved model do not match.",
                    UserWarning)

    def save_results(self, model, results):
        save_file = self.save_path / "{}/results.json".format(self.get_model_name(model))
        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)
        with open(save_file, "w") as f:
            json.dump(results, f)
    
    def get_model_name(self, model):
        name = model.name() if hasattr(model, "name") else str(model)
        return name
    
    def get_models_for_onnx_save(self, dtype=None):
        return self.training_manager.get_models_for_onnx_save(dtype=dtype)