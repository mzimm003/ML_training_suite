from ml_training_suite.training import TrainingScript
from ml_training_suite.datasets import Dataset
from ml_training_suite.models import FeatureReducer

from typing import (
    Type,
    Union,
    Any,
    Dict
)
from pathlib import Path

class FeatureReductionTraining(TrainingScript):
    def __init__(
            self,
            dataset:Union[str, Type[Dataset]] = None,
            dataset_kwargs:Dict[str, Any] = None,
            feature_reducer:Union[str, FeatureReducer] = None,
            feature_reducer_kwargs:Dict[str, Any] = None,
            save_path:Union[str, Path] = None,
            **kwargs) -> None:
        """
        Args:
            dataset: The dataset class to be used.
            dataset_kwargs: Configuration for the dataset.
            feature_reducer: The feature reducer class to be used.
            feature_reducer_kwargs: Configuration for the feature reducer.
            save_path: Specify a path to which feature reducer should be saved.
        """
        super().__init__(**kwargs)
        self.data = dataset
        self.ds_kwargs = dataset_kwargs if dataset_kwargs else {}
        self.feature_reducer = feature_reducer
        self.fr_kwargs = feature_reducer_kwargs if feature_reducer_kwargs else {}
        self.save_path = save_path
    
    def setup(self):
        self.data = Dataset.initialize(
            self.data, self.ds_kwargs)
        self.feature_reducer = FeatureReducer.initialize(
            self.feature_reducer, self.fr_kwargs)
        self.save_path = Path(self.save_path)

    def run(self):
        inp, tgt = self.data[:]
        self.feature_reducer.fit(inp, tgt)
        self.save_model(self.feature_reducer, inp)