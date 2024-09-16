from ml_training_suite.base import ML_Element
from ml_training_suite.registry import Registry

from ml_training_suite.datasets import DataHandler

class Callback(ML_Element, register=False):
    registry = Registry()
    def __init__(self) -> None:
        pass

    def on_run_begin(self, script):
        pass

    def on_fold_begin(self, script):
        pass

    def on_inference_end(self, script, data_handler:DataHandler):
        pass
    
    def on_train_batch_begin(self, script):
        pass

    def on_val_batch_begin(self, script):
        pass

    def on_model_select(self, script):
        pass

    def get_epoch_metrics(self):
        pass