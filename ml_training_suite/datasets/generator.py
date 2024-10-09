from ml_training_suite.base import ML_Element
from ml_training_suite.registry import Registry

from typing import (
    Union,
    List,
    Dict,
    Literal,
    Any
)

from pathlib import Path
import enum

import pandas as pd
import numpy as np
import h5py

class DatasetGenerator(ML_Element, register=False):
    registry = Registry()
    ROOTNAME = "database"
    DTYPE = "dtype"
    SIZE = "size"
    FILTER_FLAGS = "filter_flags"
    def __init__(
            self,
            dir:Union[str, Path],
            init_size:int,
            chunk_size:int=None,
            max_size:int=None,
            resize_step:int=None,
            filters:List[Union[int,enum.Enum]]=None,
            ) -> None:
        """
        Args:
            dir: Root directory of dataset to be generated.
        """
        super().__init__()
        self.dir = dir
        self.database_path = Path(self.dir) / self.ROOTNAME
        self.metadata = {}
        self.init_size = init_size
        self.chunk_size = chunk_size
        self.max_size = max_size
        self.resize_step = resize_step

        self.filters = [] if filters is None else [int(f) for f in filters]
        self.filters = {k: 1 << k for k in self.filters}

        self.max_filters = 32 if len(self.filters) <= 32 else 64
        if not self.resize_step is None:
            assert self.resize_step % self.chunk_size == 0

    def get_dtypes(self) -> dict:
        return {k: v[DatasetGenerator.DTYPE] for k, v in self.metadata.items()}

    def set_dtype(self, key, dtype):
        self.metadata[key][DatasetGenerator.DTYPE] = dtype

    def add_metadata_entry(self, k, v):
        dtype = np.array(v).dtype
        if 'U' in str(dtype):
            dtype = h5py.string_dtype()
        self.metadata[k] = {
            DatasetGenerator.DTYPE:dtype,
            DatasetGenerator.SIZE:np.array(v).shape}
        
    def map_dict_sample(self, sample:dict):
        for k, v in sample.items():
            if not k in self.filters:
                self.add_metadata_entry(k,v)

    def map_list_sample(self, sample:list):
        raise NotImplementedError

    def map_dataframe_sample(self, sample:pd.DataFrame):
        raise NotImplementedError

    def initialize_by_sample(self, sample):
        """
        Means of describing dataset structure by a sample.

        The sample will provide everything that will be included in each data
        point of the dataset. This allows the generator to understand structure
        needs like data dtypes, labels, etc. and for that, create a base
        skeleton.

        Args:
            sample: Single data point of dataset.
        """
        map_method = {
            dict:self.map_dict_sample,
            list:self.map_list_sample,
            pd.DataFrame:self.map_dataframe_sample,
        }
        map_method[type(sample)](sample)
        self.init_database()
    
    def add_data_by_point(self, data_point):
        pass

    def add_data_by_batch(self, data_batch):
        pass

    def init_database(self):
        raise NotImplementedError

    def resize_database(self):
        raise NotImplementedError

class HybridDatasetGenerator(DatasetGenerator):
    pass

class HDF5DatasetGenerator(DatasetGenerator):
    SPACE_AVAILABLE = 'space_available'
    CURR_SIZE = 'curr_size'
    def __init__(
            self,
            dir: Union[str, Path],
            init_size: int,
            chunk_size: int = None,
            max_size: int = None,
            resize_step:int=None,
            filters: List[Union[int,enum.Enum]] = None,
            compression:Literal["gzip","lzf","szip"] = 'gzip',
            compession_opts:Any = None,
            ) -> None:
        """
        Args:
            compression: lossless compression filter.
        """
        super().__init__(
            dir=dir,
            init_size=init_size,
            chunk_size=chunk_size,
            max_size=max_size,
            resize_step=resize_step,
            filters=filters)
        self.compression = compression
        self.compression_opts = compession_opts
        self.database_path = self.database_path.with_name(
            self.database_path.stem + '.h5')

    def init_database(self):
        if not self.database_path.parent.exists():
            self.database_path.parent.mkdir(parents=True)
        with h5py.File(self.database_path, 'w') as f:
            for key, metadata in self.metadata.items():
                f.create_dataset(
                    key,
                    shape=(self.init_size, *metadata[DatasetGenerator.SIZE]),
                    maxshape=(self.max_size, *metadata[DatasetGenerator.SIZE]),
                    dtype=metadata[DatasetGenerator.DTYPE],
                    chunks=(self.chunk_size, *metadata[DatasetGenerator.SIZE]),
                    compression=self.compression,
                    compression_opts=self.compression_opts,)
            f.create_dataset(self.FILTER_FLAGS,
                shape=(self.init_size,),
                maxshape=(self.max_size,),
                dtype='uint{}'.format(self.max_filters),
                chunks=(self.chunk_size,),
                compression=self.compression,
                compression_opts=self.compression_opts,)
            f.attrs[self.SPACE_AVAILABLE] = self.init_size
            f.attrs[self.CURR_SIZE] = self.init_size
    
    def resize_database(self, db:h5py.File, final=False):
        if final:
            for key in db.keys():
                db[key].resize(
                    db.attrs[self.CURR_SIZE] - db.attrs[self.SPACE_AVAILABLE],
                    axis=0)
            db.flush()
            db.attrs[self.CURR_SIZE] -= db.attrs[self.SPACE_AVAILABLE]
            db.attrs[self.SPACE_AVAILABLE] = 0
        else:
            for key in db.keys():
                db[key].resize(
                    db[key].shape[0] + self.resize_step,
                    axis=0)
            db.attrs[self.CURR_SIZE] += self.resize_step
            db.attrs[self.SPACE_AVAILABLE] += self.resize_step
    
    def repack(self):
        src_file = self.database_path
        dst_file = self.database_path.with_name(
            self.database_path.stem + '_repacked' + self.database_path.suffix)
        
        with h5py.File(src_file, 'r+') as f:
            self.resize_database(f, final=True)

        with h5py.File(src_file, 'r') as src, \
             h5py.File(dst_file, 'w') as dst:
                src.copy(src, dst)
        src_file.unlink()
        dst_file.rename(src_file)

    def retro_add_feature(self, db:h5py.File, key:str, value):
        self.add_metadata_entry(key, value)
        metadata = self.metadata[key]
        db.create_dataset(
            key,
            shape=(db.attrs[self.CURR_SIZE], *metadata[DatasetGenerator.SIZE]),
            maxshape=(self.max_size, *metadata[DatasetGenerator.SIZE]),
            dtype=metadata[DatasetGenerator.DTYPE],
            chunks=(self.chunk_size, *metadata[DatasetGenerator.SIZE]),
            compression=self.compression,
            compression_opts=self.compression_opts,)

    def add_data_by_point(
            self,
            data_point:dict,
            filters:List[Union[int,enum.Enum]]=None):
        with h5py.File(self.database_path, 'r+') as f:
            if f.attrs[self.SPACE_AVAILABLE] == 0:
                self.resize_database(f)

            index = f.attrs[self.CURR_SIZE] - f.attrs[self.SPACE_AVAILABLE]
            for key, value in data_point.items():
                if not key in f:
                    self.retro_add_feature(f, key, value)
                f[key][index] = value
            key = self.FILTER_FLAGS
            value = sum(self.filters[int(k)] for k in filters)
            f[key][index] = value
            f.attrs[self.SPACE_AVAILABLE] -= 1

    def add_data_by_batch(
            self,
            data_batch:dict,
            filters:List[List[Union[int,enum.Enum]]]=None):
        batch_len = len(next(iter(data_batch.values())))
        with h5py.File(self.database_path, 'r+') as f:
            while f.attrs[self.SPACE_AVAILABLE] < batch_len:
                self.resize_database(f)

            index_start = f.attrs[self.CURR_SIZE] - f.attrs[self.SPACE_AVAILABLE]
            index_slice = slice(index_start, index_start + batch_len)
            for key, value in data_batch.items():
                if not key in f:
                    self.retro_add_feature(f, key, value[0])
                f[key][index_slice] = value
            key = self.FILTER_FLAGS
            value = [sum(self.filters[int(k)] for k in filt) for filt in filters]
            f[key][index_slice] = value
            f.attrs[self.SPACE_AVAILABLE] -= batch_len

class PandasDatasetGenerator(DatasetGenerator):
    pass