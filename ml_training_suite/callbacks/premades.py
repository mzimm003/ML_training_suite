from ml_training_suite.callbacks import Metric, Callback

import numpy as np

from typing import List, Dict

class SupervisedTrainingCallback(Callback):
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
        train_mets = np.empty_like(self.training_metrics)
        for fold, fold_metrics in enumerate(self.training_metrics):
            for mod, mod_metrics in enumerate(fold_metrics):
                for i, met in enumerate(mod_metrics):
                    res = met.get_result()
                    ret["fold{}_model{}_train_{}".format(met.fold_num,met.model_name,met)] = res
                    train_mets[fold][mod][i] = res
                    
        val_mets = np.empty_like(self.validation_metrics)
        for fold, fold_metrics in enumerate(self.validation_metrics):
            for mod, mod_metrics in enumerate(fold_metrics):
                for i, met in enumerate(mod_metrics):
                    res = met.get_result()
                    ret["fold{}_model{}_val_{}".format(met.fold_num,met.model_name,met)] = res
                    val_mets[fold][mod][i] = res
                    
        for fold in range(len(self.training_metrics)):
            for i, met in enumerate(self.TRAINING_METRICS):
                ret["mean_fold{}_train_{}".format(fold, met.__name__)] = np.mean(train_mets[fold,:,i])
            for i, met in enumerate(self.VALIDATION_METRICS):
                ret["mean_fold{}_val_{}".format(fold, met.__name__)] = np.mean(val_mets[fold,:,i])

        for mod_idx, mod_name in enumerate(self.models_in_training):
            for i, met in enumerate(self.TRAINING_METRICS):
                ret["mean_model{}_train_{}".format(mod_name, met.__name__)] = np.mean(train_mets[:,mod_idx,i])
            for i, met in enumerate(self.VALIDATION_METRICS):
                ret["mean_model{}_val_{}".format(mod_name, met.__name__)] = np.mean(val_mets[:,mod_idx,i])
        
        for i, met in enumerate(self.TRAINING_METRICS):
            ret["mean_train_{}".format(met.__name__)] = np.mean(train_mets[:,:,i])
        for i, met in enumerate(self.VALIDATION_METRICS):
            ret["mean_val_{}".format(met.__name__)] = np.mean(val_mets[:,:,i])
        return ret
    
    def on_run_begin(self, script):
        training_manager = script.training_manager
        tracker = training_manager.training_tracker
        num_parent_models_per_fold = len(training_manager.models)/len(tracker.folds)
        num_training_models_per_fold = len(tracker.getTrainingModelsByFold(0))
        num_training_models_per_parent = num_training_models_per_fold/num_parent_models_per_fold
        self.models_in_training = [
            '{:d}{}'.format(
                int(i//num_training_models_per_parent),
                chr(65+int(i%num_training_models_per_parent)))
            for i in
            range(num_training_models_per_fold)]
        
        self.training_metrics:List[List[List[Metric]]] = []
        self.training_metrics_model_mapping:Dict[int, List[Metric]] = {}
        for i in tracker.folds.keys():
            self.training_metrics.append([])
            models_in_training_ids = tracker.getTrainingModelsByFold(i)
            for j, (mod_name, mod_id) in enumerate(zip(self.models_in_training, models_in_training_ids)):
                self.training_metrics[i].append([])
                for met in SupervisedTrainingCallback.TRAINING_METRICS:
                    met_obj = met(mod_name, j)
                    self.training_metrics[i][j].append(met_obj)
                self.training_metrics_model_mapping[mod_id] = self.training_metrics[i][j]

        self.validation_metrics:List[List[List[Metric]]] = []
        self.validation_metrics_model_mapping:Dict[int, List[Metric]] = {}
        for i in tracker.folds.keys():
            self.validation_metrics.append([])
            models_in_training_ids = tracker.getTrainingModelsByFold(i)
            for j, (mod_name, mod_id) in enumerate(zip(self.models_in_training, models_in_training_ids)):
                self.validation_metrics[i].append([])
                for met in SupervisedTrainingCallback.VALIDATION_METRICS:
                    met_obj = met(mod_name, j)
                    self.validation_metrics[i][j].append(met_obj)
                self.validation_metrics_model_mapping[mod_id] = self.validation_metrics[i][j]
        self.inference_mode = SupervisedTrainingCallback.INFERENCE_MODES[0]
    
    def on_train_batch_begin(self, script):
        self.inference_mode = SupervisedTrainingCallback.INFERENCE_MODES[0]

    def on_val_batch_begin(self, script):
        self.inference_mode = SupervisedTrainingCallback.INFERENCE_MODES[1]

    def on_inference_end(self, script, trainer, data_handler):
        metric_set = (self.training_metrics_model_mapping
                      if self.inference_mode == SupervisedTrainingCallback.INFERENCE_MODES[0]
                      else self.validation_metrics_model_mapping)
        for met in metric_set[id(trainer.model)]:
            met.include(data_handler=data_handler)