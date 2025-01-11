from ml_training_suite.base import ML_Element
from ml_training_suite.registry import Registry
from ml_training_suite.models import Model

import torch
from torch import nn
from torch.utils.data import Dataset as DatasetTorch
from torch.utils.data import Subset as SubsetTorch

import numpy as np
from numpy.typing import ArrayLike

from typing import (
    Union,
    Tuple,
    List,
    Callable,
    Iterable,
    Any
)
from pathlib import Path
import math
import numbers

class MemSafeAttr:
    """
    To prevent memory problems with multiprocessing, class provides the utility 
    functions:

    * strings_to_mem_safe_val_and_offset
    * mem_safe_val_and_offset_to_string
    * string_to_sequence
    * sequence_to_string
    * pack_sequences
    * unpack_sequence

    Provided by https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519.
    These should be used in lieu of lists or dicts in the data retrieval process
    (e.g. for data labels).

    See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662 
    for a summary of the issue.
    """
    def __init__(self, attr:List[str]): #TODO, expand to general lists and dicts
        self.dtype = type(next(iter(attr)))
        val, offset = self.to_mem_safe_val_and_offset(
            attr=attr)
        self.val = val
        self.offset = offset

    def __getitem__(self, key:int):
        return self.mem_safe_val_and_offset_to(
            self.val,
            self.offset,
            key
        )

    def to_mem_safe_val_and_offset(self, attr: List[Union[str,int]]) -> Tuple[np.ndarray,np.ndarray]:
        """
        Utility function.
        """
        method_type = self.dtype
        if issubclass(method_type, numbers.Number):
            method_type = numbers.Number
        return {
            str:MemSafeAttr.strings_to_mem_safe_val_and_offset,
            numbers.Number:MemSafeAttr.num_to_mem_safe_val_and_offset,
        }[method_type](attr)

    def mem_safe_val_and_offset_to(self, v, o, index:int) -> Tuple[np.ndarray,np.ndarray]:
        """
        Utility function.
        """
        method_type = self.dtype
        if issubclass(method_type, numbers.Number):
            method_type = numbers.Number
        return {
            str:MemSafeAttr.mem_safe_val_and_offset_to_string,
            numbers.Number:MemSafeAttr.mem_safe_val_and_offset_to_num,
        }[method_type](v, o, index)

    # --- UTILITY FUNCTIONS ---    
    @staticmethod
    def num_to_mem_safe_val_and_offset(ints: List[str]) -> Tuple[np.ndarray,np.ndarray]:
        """
        Utility function.
        """
        val = np.array(ints)
        offset = None
        return val, offset
    
    @staticmethod
    def mem_safe_val_and_offset_to_num(v, o, index:int) -> Tuple[np.ndarray,np.ndarray]:
        '''
        Utility function.
        '''
        return v[index]
    
    @staticmethod
    def strings_to_mem_safe_val_and_offset(strings: List[str]) -> Tuple[np.ndarray,np.ndarray]:
        """
        Utility function.
        """
        seqs = [MemSafeAttr.string_to_sequence(s) for s in strings]
        return MemSafeAttr.pack_sequences(seqs)
    
    @staticmethod
    def mem_safe_val_and_offset_to_string(v, o, index:int) -> Tuple[np.ndarray,np.ndarray]:
        '''
        Utility function.
        '''
        seq = MemSafeAttr.unpack_sequence(v, o, index)
        return MemSafeAttr.sequence_to_string(seq)
    
    @staticmethod
    def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
        """
        Utility function.
        """
        return np.array([ord(c) for c in s], dtype=dtype)

    @staticmethod
    def sequence_to_string(seq: np.ndarray) -> str:
        """
        Utility function.
        """
        return ''.join([chr(c) for c in seq])

    @staticmethod
    def pack_sequences(seqs: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Utility function.
        """
        values = np.concatenate(seqs, axis=0)
        offsets = np.cumsum([len(s) for s in seqs])
        return values, offsets

    @staticmethod
    def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
        """
        Utility function.
        """
        off1 = offsets[index]
        if index > 0:
            off0 = offsets[index - 1]
        elif index == 0:
            off0 = 0
        else:
            raise ValueError(index)
        return values[off0:off1]

class Batch:
    """
    Basic batching scheme.

    Provided by https://pytorch.org/docs/stable/data.html. Providing attribute
    descriptions for the class is helpful for the trainer data handler mapping.
    See ml_training_suite.training.
    """
    FET="fet"
    TGT="tgt"
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return Batch(batch)

class Dataset(DatasetTorch, ML_Element, register=False):
    """
    A basis for database creation and dataset serving.

    For subclasses, the batch class and collate_wrapper should be reviewed
    and revised (subclassed/overwritten) as necessary if the basic method
    supplied does not suffice.
    """
    registry = Registry()
    collate_wrapper = collate_wrapper
    labels:Union[None, torch.Tensor]
    AUTONAME = "complete_generated_dataset"
    LBL_FILE_COUNT = "total_label_files"
    LBL_COUNT = "total_labels"
    CURR_FILE_IDX = "current_file"
    LBL_COUNT_BY_FILE = "_label_counts"
    LBL_BINS = "_cum_label_counts"
    def __init__(
            self,
            seed:int=None,
            ) -> None:
        super().__init__()
        self.seed = seed

    def getName(self):
        return self.__class__.__name__
    
    def __setattr__(self, name: str, value: Any) -> None:
        #TODO implement to automatically convert dict and list attributes to
        # necessary split value and offset attributes stored as numpy arrays by
        # utility funtions below.
        if type(value) in [list, dict]:
            if not value in [[],{}]:
                value = MemSafeAttr(value)
        super().__setattr__(name, value)

class Subset(SubsetTorch):
    dataset:Dataset
    def getLabels(self):
        return self.dataset.labels[self.indices]

class DataHandlerGenerator:
    pipeline:List[Tuple[str, Callable]]
    def __init__(
            self,
            **inputs):
        self.pipeline = []
        for key, inp in inputs.items():
            setattr(self, key, inp)
        for i in range(len(self.pipeline)):
            inps = self.pipeline[i][0]
            mod = self.pipeline[i][1]
            if not isinstance(self.pipeline[i][0],list):
                assert isinstance(self.pipeline[i][0],str)
                inps = [self.pipeline[i][0]]
            if isinstance(self.pipeline[i][0],str) or isinstance(self.pipeline[i][0],Path):
                mod = Model.load_model(mod)
            self.pipeline[i] = (inps, mod)
    def get_data_handler(self, data):
        inputs = {
            inp:getattr(data, val)
            for inp, val in self.__dict__.items() if inp != "pipeline"
        }
        return DataHandler(
            pipeline=self.pipeline,
            **inputs)

class DataHandler:
    loss:torch.Tensor
    output:torch.Tensor
    aux_output:Union[Tuple[torch.Tensor],None]
    target:torch.Tensor
    last_lr:float
    pipeline:List[Tuple[str, Callable]]
    def __init__(
            self,
            target=None,
            pipeline=None,
            **inputs,
            ):
        self.pipeline = pipeline
        self.target = target
        self.inputs = inputs

    def process_data(self, model:nn.Module, dtype=None, trunc_batch=None):
        param_ref = next(iter(model.parameters()))
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                if dtype:
                    value = value.to(dtype=dtype)
                if trunc_batch:
                    value = value[:trunc_batch,...]
                setattr(self, key, value.to(device=param_ref.device))
            elif key == 'inputs' and isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        if dtype:
                            v = v.to(dtype=dtype)
                        if trunc_batch:
                            v = v[:trunc_batch,...]
                        self.inputs[k] = v.to(device=param_ref.device)
        self.run_pipeline()
    
    def get_inputs(self):
        return self.inputs
    
    def set_last_lr(self, last_lr):
        self.last_lr = last_lr

    def get_last_lr(self):
        return self.last_lr

    def set_loss(self, criterion:nn.Module):
        kwargs = {"input":self.output,
                  "target":self.target}
        if self.aux_output:
            kwargs["aux_inputs"] = self.aux_output
        loss = criterion(**kwargs)
        self.loss = loss

    def set_model_output(self, output):
        self.aux_output = None
        if type(output) in [list, tuple]:
            self.aux_output = output[1:]
        else:
            output = output,
        self.output = output[0]

    def run_pipeline(self):
        # Pipeline exists to modify input data prior to providing to training
        # model, e.g. piping through a feature reduction model. To that end, 
        # it is assumed inputs of the pipeline should be replaced by the
        # outputs.
        for inp, mod in self.pipeline:
            inp = {k:self.inputs[k] for k in inp}
            mod.to(device=next(iter(inp.values())).device)
            if len(inp) > 1:
                raise NotImplementedError
            out, *_ = mod(**inp)
            self.inputs[list(inp.keys())[0]] = out

class ClusterManager:
    def __init__(self):
        self.train_clusts = None
        self.test_clusts = None

    def preserve_cluster_split(
            self,
            idxs:list[int],
            clusters:list[int],
            train_size:int,
            test_size:int,
            ):
        # Determine clusters that exist, in what position, and in what size
        unq_clusts, unq_clust_first_indx, unq_clust_counts = np.unique(
            clusters, return_index=True, return_counts=True)
        clust_order_idxs = np.argsort(unq_clust_first_indx)
        clusts_ordered = unq_clusts[clust_order_idxs]
        unq_clust_counts_ordered = unq_clust_counts[clust_order_idxs]
        if self.train_clusts is None:
            # Establish train clusters by first clusters to fit
            valid_train_clusts_mask = unq_clust_counts_ordered <= train_size
            cum_unq_clust_counts_ordered = np.cumsum(
                unq_clust_counts_ordered[valid_train_clusts_mask])
            fits_train_clusts_mask = cum_unq_clust_counts_ordered <= train_size            
            self.train_clusts = clusts_ordered[valid_train_clusts_mask][fits_train_clusts_mask]
            
            # Establish test clusters by first clusters to fit not in train set
            test_size += train_size - cum_unq_clust_counts_ordered[fits_train_clusts_mask][-1]
            valid_test_clusts_mask = np.logical_and(
                unq_clust_counts_ordered <= test_size,
                np.logical_not(np.isin(clusts_ordered, self.train_clusts))
                )
            cum_unq_clust_counts_ordered = np.cumsum(
                unq_clust_counts_ordered[valid_test_clusts_mask])
            fits_test_clusts_mask = cum_unq_clust_counts_ordered <= test_size
            self.test_clusts = clusts_ordered[valid_test_clusts_mask][fits_test_clusts_mask]
        else:
            already_train_mask = np.isin(clusts_ordered, self.train_clusts)
            already_test_mask = np.isin(clusts_ordered, self.test_clusts)
            not_already_placed_mask = np.logical_not(np.logical_or(
                already_train_mask,
                already_test_mask
            ))
            if not_already_placed_mask.any():
                new_cm = ClusterManager()
                new_cm.preserve_cluster_split(
                    idxs=idxs[np.isin(clusters, clusts_ordered[not_already_placed_mask])],
                    clusters = clusters[np.isin(clusters, clusts_ordered[not_already_placed_mask])],
                    train_size= train_size - np.sum(unq_clust_counts_ordered[already_train_mask]),
                    test_size = test_size - np.sum(unq_clust_counts_ordered[already_test_mask]),)
                self.train_clusts = np.append(self.train_clusts, new_cm.train_clusts)
                self.test_clusts = np.append(self.test_clusts, new_cm.test_clusts)
        train_idxs = idxs[np.isin(clusters,self.train_clusts)]
        test_idxs = idxs[np.isin(clusters, self.test_clusts)]

        return train_idxs, test_idxs

def train_test_split(
        *arrays:ArrayLike,
        train_size:Union[int,float]=None,
        test_size:Union[int, float]=None,
        shuffle:bool=True,
        clusters:Union[List[ArrayLike],None]=None,
        stratify:Union[List[ArrayLike],None]=None,
        seed=None,
        ):
    """
    Imitates scikit-learn function with the same name.

    Created to provide a speed up, especially when stratifying.

    WARNING: Use of clusters is a subset sum problem (i.e. NP-Hard) when solved
    for train/test sizes precisely, which, for large datasets, may be
    inneficient to calculate. So, instead the train_size serves as a maximum,
    and the more granular and uniform cluster sizes are, the closer to the
    maximum the train set will be in practice. The test set will then consider
    any difference from the train set size and its maximum combined with
    test_size to be the maximum size.

    WARNING-2: Use of clusters and stratification together may force poor
    stratificaiton. Clusters will be prioritized, as its purpose is to prevent
    data leakage. Due to the NP-Hard nature of the problem, and the approximate
    solution implemented, it's possible the stratification of one class will
    force all examples of another class into either the train or test set based
    on their clustered relationship to the class already stratified.

    Args:
        arrays: Data(s) to be split.
        train_size: Number or proportion of data points to be split into train
          set.
        test_size: Number or proportion of data points to be split into test
          set.
        shuffle: Whether to shuffle data before splitting.
        clusters: Group labels where all members of a group should be in the
          same split.
        stratify: Group labels where members of a group should be evenly split.
        seed: If provided, will ensure consistent results for RNG operations.
    """
    assert isinstance(test_size, type(train_size))
    rng = np.random.default_rng(seed)
    train_idxs = test_idxs = None
    for i, array in enumerate(arrays):
        assert stratify is None or len(stratify[i]) == len(array)
        idxs = np.arange(len(array))

        if shuffle:
            idxs = rng.permutation(idxs)

        if isinstance(test_size, float):
            assert test_size + train_size <= 1.
            train_size = int(len(array)*train_size)
            test_size = int(math.ceil(len(array)*test_size))
        
        assert (train_size + test_size) <= len(array)

        if stratify is None:
            if clusters is None:
                train_idxs = idxs[:train_size]
                test_idxs = idxs[train_size:train_size+test_size]
            else:
                train_idxs, test_idxs = ClusterManager().preserve_cluster_split(
                    idxs=idxs,
                    clusters=clusters[i][idxs],
                    train_size=train_size,
                    test_size=test_size,
                )
        else:
            train_idxs = []
            test_idxs = []
            strat = stratify[i][idxs]
            class_id, class_count = np.unique(strat, return_counts=True)
            class_ratios = class_count/len(strat)
            cm = ClusterManager()
            for c, rat in zip(class_id, class_ratios):
                mask = strat == c
                population = idxs[mask]
                train_sample_size = int(train_size*rat)
                test_sample_size = int(math.ceil(test_size*rat))
                if clusters is None:
                    train_idxs.extend(
                        population[:train_sample_size])
                    test_idxs.extend(
                        population[train_sample_size:train_sample_size+test_sample_size])
                else:
                    train_idxs_add, test_idxs_add = cm.preserve_cluster_split(
                        idxs=population,
                        clusters=clusters[i][population],
                        train_size=train_sample_size,
                        test_size=test_sample_size
                    )
                    train_idxs.extend(
                        train_idxs_add)
                    test_idxs.extend(
                        test_idxs_add)
            train_idxs = rng.permutation(train_idxs)
            test_idxs = rng.permutation(test_idxs)
            
        yield train_idxs, test_idxs