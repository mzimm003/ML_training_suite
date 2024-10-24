from ml_training_suite.base import ML_Element
from ml_training_suite.registry import Registry

class Callback(ML_Element, register=False):
    registry = Registry()
    def __init__(self, TRAINING_METRICS, VALIDATION_METRICS) -> None:
        self.TRAINING_METRICS = TRAINING_METRICS
        self.VALIDATION_METRICS = VALIDATION_METRICS

    def on_run_begin(self, script):
        pass

    def on_fold_begin(self, script):
        pass

    def on_inference_end(self, script, trainer, data_handler):
        pass
    
    def on_train_batch_begin(self, script):
        pass

    def on_val_batch_begin(self, script):
        pass

    def get_epoch_metrics(self):
        pass