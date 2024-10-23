from ml_training_suite.training import (
    TrainingScript,
    TrainingManager,
    Trainer,
    Optimizer,
    LRScheduler,
    Criterion)
from ml_training_suite.datasets import Dataset
from ml_training_suite.models import Model
from ml_training_suite.registry import IterOptional
from ml_training_suite.callbacks import Callback

from typing import (
    Type,
    Union,
    Any,
    Callable,
    List,
    Tuple,
    Dict
)
from pathlib import Path

import torch
from torch import nn


class SupervisedTraining(TrainingScript):
    def __init__(
            self,
            dataset:Union[str, Type[Dataset]] = None,
            dataset_kwargs:Dict[str, Any] = None,
            pipelines:IterOptional[List[Tuple[str, Callable]]] = None,
            trainer_class:Type[Trainer] = None,
            models:IterOptional[Union[str, Model]] = None,
            models_kwargs:IterOptional[Dict[str, Any]] = None,
            optimizers:IterOptional[Type[Optimizer]]= None,
            optimizers_kwargs:IterOptional[Dict[str, Any]] = None,
            lr_schedulers:IterOptional[Type[LRScheduler]]= None,
            lr_schedulers_kwargs:IterOptional[Dict[str, Any]] = None,
            criterion:Union[Type[Criterion], Type[torch.nn.modules.loss._Loss]] = None,
            criterion_kwargs:Dict[str, Any] = None,
            save_path:Union[str, Path] = None,
            balance_training_set = False,
            k_fold_splits:int = 4,
            batch_size:int = 64,
            shuffle:bool = True,
            num_workers:int = 0,
            callback:Type[Callback] = None,
            callback_kwargs:Dict[str, Any] = None,
            save_torch:bool = True,
            save_onnx:bool = True,
            **kwargs) -> None:
        """
        Args:
            dataset: The dataset class to be used.
            dataset_kwargs: Configuration for the dataset.
            pipelines: Models to pass data through prior to training model.
            trainer_class: Class governing model training.
            classifiers: The classifier class to be used.
            classifiers_kwargs: Configuration for the classifier.
            optimizers: The optimizer class to be used.
            optimizers_kwargs : Configuration for the optimizer.
            lr_schedulers: The learning rate scheduler class to be used, if any.
            lr_schedulers_kwargs : Configuration for the learning rate scheduler.
            criterion: The criterion class to be used.
            criterion_kwargs : Configuration for the criterion.
            save_path: Specify a path to which classifier should be saved.
            balance_training_set: Create a sampling scheme so that each class
              is trained on equally often.
            k_fold_splits: Number of folds used for dataset in training.
            batch_size: Number of data points included in training batches.
            shuffle: Whether to shuffle data points.
            num_workers: For multiprocessing.
            callback: Class of processes interjected into training run.
            callback_kwargs: Configuration for callback.
        """
        super().__init__(
            save_torch=save_torch,
            save_onnx=save_onnx,
            **kwargs)
        self.data = dataset
        self.ds_kwargs = dataset_kwargs if dataset_kwargs else {}
        self.balance_training_set = balance_training_set
        self.k_fold_splits = k_fold_splits
        self.training_manager = None
        self.trainer = Trainer if trainer_class is None else trainer_class
        self.dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.data.collate_wrapper,
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None
            )

        self.pipelines = pipelines
        self.classifiers:IterOptional[nn.Module] = models
        if not isinstance(self.classifiers, list):
            self.classifiers = [self.classifiers]
        self.classifiers_kwargs = models_kwargs if models_kwargs else [{}]*len(self.classifiers)
        if not isinstance(self.classifiers_kwargs, list):
            self.classifiers_kwargs = [self.classifiers_kwargs]*len(self.classifiers)
        self.optimizers = optimizers
        if not isinstance(self.optimizers, list):
            self.optimizers = [self.optimizers]*len(self.classifiers)
        self.optimizers_kwargs = optimizers_kwargs if optimizers_kwargs else [{}]*len(self.classifiers)
        if not isinstance(self.optimizers_kwargs, list):
            self.optimizers_kwargs = [self.optimizers_kwargs]*len(self.classifiers)
        self.lr_schedulers = lr_schedulers
        if not isinstance(self.optimizers, list):
            self.lr_schedulers = [self.lr_schedulers]*len(self.classifiers)
        self.lr_schedulers_kwargs = lr_schedulers_kwargs if lr_schedulers_kwargs else [{}]*len(self.classifiers)
        if not isinstance(self.lr_schedulers_kwargs, list):
            self.lr_schedulers_kwargs = [self.lr_schedulers_kwargs]*len(self.classifiers)
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs if criterion_kwargs else {}
        self.save_path = save_path
        self.callback = callback if callback else Callback.SupervisedTrainingCallback
        self.callback_kwargs = callback_kwargs if callback_kwargs else {}

    def setup(self):
        self.callback = Callback.initialize(
            self.callback, self.callback_kwargs)
        self.data = Dataset.initialize(
            self.data, self.ds_kwargs)
        print("Dataset initialized.")
        self.training_manager = TrainingManager(
            data=self.data,
            dl_kwargs=self.dl_kwargs,
            num_splits=self.k_fold_splits,
            balance_training_set=self.balance_training_set,
            shuffle=self.dl_kwargs["shuffle"],
            trainer_class=self.trainer,
            pipelines=self.pipelines,
            models=self.classifiers,
            models_kwargs=self.classifiers_kwargs,
            optimizers=self.optimizers,
            optimizers_kwargs=self.optimizers_kwargs,
            lr_schedulers=self.lr_schedulers,
            lr_schedulers_kwargs=self.lr_schedulers_kwargs,
            criterion=self.criterion,
            criterion_kwargs=self.criterion_kwargs,
        )
        print("Training manager initialized.")
        self.save_path = Path(self.save_path)
        print("Completed setup.")

    def run(self):
        self.callback.on_run_begin(self)
        for train_elements in self.training_manager:
            self.callback.on_fold_begin(self)
            for trainer in train_elements.trainers:
                trainer.model.train()
            for data in train_elements.train_data:
                self.callback.on_train_batch_begin(self)
                # results = None
                # with Pool(ray_address="auto") as pool:
                #     results = pool.map(
                #         self.train,
                #         zip(
                #             train_elements.trainers,
                #             [data]*len(train_elements.trainers)))
                # for res in results:
                #     self.callback.on_model_select(self)
                #     self.callback.on_inference_end(
                #         self,
                #         data_handler=res)
                    
                for trainer in train_elements.trainers:
                    self.callback.on_inference_end(
                        self,
                        trainer=trainer,
                        data_handler=trainer.infer_itr(data))
                
            for trainer in train_elements.trainers:
                trainer.model.eval()
            for data in train_elements.val_data:
                self.callback.on_val_batch_begin(self)
                # results = None
                # with Pool(ray_address="auto") as pool:
                #     results = pool.map(
                #         self.train,
                #         zip(
                #             train_elements.trainers,
                #             [data]*len(train_elements.trainers)))
                # for res in results:
                #     self.callback.on_model_select(self)
                #     self.callback.on_inference_end(
                #         self,
                #         data_handler=res)
                for trainer in train_elements.trainers:
                    self.callback.on_inference_end(
                        self,
                        trainer=trainer,
                        data_handler=trainer.infer_itr(data))
        self.training_manager.step_lr_schedulers()
        return self.callback.get_epoch_metrics()