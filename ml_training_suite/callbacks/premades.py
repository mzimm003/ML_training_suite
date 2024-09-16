from ml_training_suite.callbacks import Metric, Callback
from ml_training_suite.datasets import DataHandler

import numpy as np

class ClassifierTrainingCallback(Callback):
    TRAINING_METRICS = [
        Metric.Loss,
        Metric.LearningRate]
    VALIDATION_METRICS = [
        Metric.Loss,
        Metric.Accuracy,
        Metric.Precision,
        Metric.Recall,
        Metric.pAUC,
        Metric.LearningRate]
    INFERENCE_MODES = ["training","validation"]
    
    def get_epoch_metrics(self):
        ret = {}
        def get_res(x:Metric):
            return x.get_result()
        get_res = np.vectorize(get_res)
        train_mets = get_res(self.training_metrics)
        for f in self.folds:
            for mod in self.models:
                for i, met in enumerate(self.training_metrics[f][mod]):
                    res = train_mets[f][mod][i]
                    ret["fold{}_model{}_train_{}".format(f,mod,met)] = res
        val_mets = get_res(self.validation_metrics)
        for f in self.folds:
            for mod in self.models:
                for i, met in enumerate(self.validation_metrics[f][mod]):
                    res = val_mets[f][mod][i]
                    ret["fold{}_model{}_val_{}".format(f,mod,met)] = res
        
        for f in self.folds:
            for i, met in enumerate(self.TRAINING_METRICS):
                ret["mean_fold{}_train_{}".format(f, met.__name__)] = np.mean(train_mets[f,:,i])
            for i, met in enumerate(self.VALIDATION_METRICS):
                ret["mean_fold{}_val_{}".format(f, met.__name__)] = np.mean(val_mets[f,:,i])

        for mod in self.models:
            for i, met in enumerate(self.TRAINING_METRICS):
                ret["mean_model{}_train_{}".format(f, met.__name__)] = np.mean(train_mets[:,mod,i])
            for i, met in enumerate(self.VALIDATION_METRICS):
                ret["mean_model{}_val_{}".format(f, met.__name__)] = np.mean(val_mets[:,mod,i])
        
        for i, met in enumerate(self.TRAINING_METRICS):
            ret["mean_train_{}".format(met.__name__)] = np.mean(train_mets[:,:,i])
        for i, met in enumerate(self.VALIDATION_METRICS):
            ret["mean_val_{}".format(met.__name__)] = np.mean(val_mets[:,:,i])
        return ret
    
    def on_run_begin(self, script):
        training_manager = script.training_manager
        self.models = range(len(training_manager[0].trainers))
        self.folds = range(len(training_manager))
        self.training_metrics:list[list[list[Metric]]] = [[[
            met(i, j) for met in ClassifierTrainingCallback.TRAINING_METRICS]
            for i in self.models]
            for j in self.folds]
        self.validation_metrics:list[list[list[Metric]]] = [[[
            met(i, j) for met in ClassifierTrainingCallback.VALIDATION_METRICS]
            for i in self.models]
            for j in self.folds]
        self.model = -1
        self.fold = -1
        self.inference_mode = ClassifierTrainingCallback.INFERENCE_MODES[0]

    def on_fold_begin(self, script):
        self.fold += 1
    
    def on_model_select(self, script):
        self.model += 1
    
    def on_train_batch_begin(self, script):
        self.model = -1
        self.inference_mode = ClassifierTrainingCallback.INFERENCE_MODES[0]

    def on_val_batch_begin(self, script):
        self.model = -1
        self.inference_mode = ClassifierTrainingCallback.INFERENCE_MODES[1]

    def on_inference_end(self, script, data_handler:DataHandler):
        metric_set = (self.training_metrics
                      if self.inference_mode == ClassifierTrainingCallback.INFERENCE_MODES[0]
                      else self.validation_metrics)
        for met in metric_set[self.fold][self.model]:
            met.include(data_handler=data_handler)

class LRRangeTestCallback(Callback):
    TRAINING_METRICS = [
        Metric.Loss,
        Metric.LearningRate]
    VALIDATION_METRICS = [
        Metric.Loss,
        Metric.Accuracy,
        Metric.Precision,
        Metric.Recall,
        Metric.pAUC,
        Metric.LearningRate]
    INFERENCE_MODES = ["training","validation"]
    
    def get_epoch_metrics(self):
        ret = {}
        def get_res(x:Metric):
            return x.get_result()
        get_res = np.vectorize(get_res)
        train_mets = get_res(self.training_metrics)
        for f in self.folds:
            for mod in self.models:
                for i, met in enumerate(self.training_metrics[f][mod]):
                    res = train_mets[f][mod][i]
                    ret["fold{}_model{}_train_{}".format(f,mod,met)] = res
        val_mets = get_res(self.validation_metrics)
        for f in self.folds:
            for mod in self.models:
                for i, met in enumerate(self.validation_metrics[f][mod]):
                    res = val_mets[f][mod][i]
                    ret["fold{}_model{}_val_{}".format(f,mod,met)] = res
        
        for f in self.folds:
            for i, met in enumerate(self.TRAINING_METRICS):
                ret["mean_fold{}_train_{}".format(f, met.__name__)] = np.mean(train_mets[f,:,i])
            for i, met in enumerate(self.VALIDATION_METRICS):
                ret["mean_fold{}_val_{}".format(f, met.__name__)] = np.mean(val_mets[f,:,i])

        for mod in self.models:
            for i, met in enumerate(self.TRAINING_METRICS):
                ret["mean_model{}_train_{}".format(f, met.__name__)] = np.mean(train_mets[:,mod,i])
            for i, met in enumerate(self.VALIDATION_METRICS):
                ret["mean_model{}_val_{}".format(f, met.__name__)] = np.mean(val_mets[:,mod,i])
        
        for i, met in enumerate(self.TRAINING_METRICS):
            ret["mean_train_{}".format(met.__name__)] = np.mean(train_mets[:,:,i])
        for i, met in enumerate(self.VALIDATION_METRICS):
            ret["mean_val_{}".format(met.__name__)] = np.mean(val_mets[:,:,i])
        return ret
    
    def on_run_begin(self, script):
        training_manager = script.training_manager
        self.models = range(len(training_manager[0].trainers))
        self.folds = range(len(training_manager))
        self.training_metrics:list[list[list[Metric]]] = [[[
            met(i, j) for met in LRRangeTestCallback.TRAINING_METRICS]
            for i in self.models]
            for j in self.folds]
        self.validation_metrics:list[list[list[Metric]]] = [[[
            met(i, j) for met in LRRangeTestCallback.VALIDATION_METRICS]
            for i in self.models]
            for j in self.folds]
        self.model = -1
        self.fold = -1
        self.inference_mode = LRRangeTestCallback.INFERENCE_MODES[0]

    def on_fold_begin(self, script):
        self.fold += 1
    
    def on_model_select(self, script):
        self.model += 1
    
    def on_train_batch_begin(self, script):
        self.model = -1
        self.inference_mode = LRRangeTestCallback.INFERENCE_MODES[0]

    def on_val_batch_begin(self, script):
        self.model = -1
        self.inference_mode = LRRangeTestCallback.INFERENCE_MODES[1]

    def on_inference_end(self, script, data_handler:DataHandler):
        metric_set = (self.training_metrics
                      if self.inference_mode == LRRangeTestCallback.INFERENCE_MODES[0]
                      else self.validation_metrics)
        for met in metric_set[self.fold][self.model]:
            met.include(data_handler=data_handler)
