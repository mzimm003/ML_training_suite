from ml_training_suite.callbacks import Metric, Callback

from functools import partial

import numpy as np

from typing import List, Dict, Union

class SupervisedTrainingCallback(Callback):
    INFERENCE_MODES = ["training","validation"]

    def __init__(
            self,
            pauc_class_idxs:Union[int,List[int]]=-1,
            target_label_probability_dim:Union[int, bool]=False,
            model_label_probability_dim:Union[int, bool]=False) -> None:
        """
        Gathers metrics for supervised training.

        If training is for categorical labels, supplying the probability
        dimensions will prompt metrics to take the maximum liklihood over
        labels when determining things like accuracy.

        Args:
            target_label_probability_dim: Dimension representing label distribution.
            model_label_probability_dim: Dimension representing label distribution.
        """
        if not isinstance(pauc_class_idxs, list):
            pauc_class_idxs = [pauc_class_idxs]
        super().__init__(
            TRAINING_METRICS=[
                Metric.Loss,
                Metric.LearningRate],
            VALIDATION_METRICS=[
                Metric.Loss,
                Metric.Accuracy,
                Metric.Precision,
                Metric.Recall,
                Metric.LearningRate,
                *[partial(Metric.pAUC, class_idx=i) for i in pauc_class_idxs],
                ],)
        self.target_label_probability_dim = target_label_probability_dim
        self.model_label_probability_dim = model_label_probability_dim
    
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
                for met in self.TRAINING_METRICS:
                    met_obj = met(mod_name, j, self)
                    self.training_metrics[i][j].append(met_obj)
                self.training_metrics_model_mapping[mod_id] = self.training_metrics[i][j]

        self.validation_metrics:List[List[List[Metric]]] = []
        self.validation_metrics_model_mapping:Dict[int, List[Metric]] = {}
        for i in tracker.folds.keys():
            self.validation_metrics.append([])
            models_in_training_ids = tracker.getTrainingModelsByFold(i)
            for j, (mod_name, mod_id) in enumerate(zip(self.models_in_training, models_in_training_ids)):
                self.validation_metrics[i].append([])
                for met in self.VALIDATION_METRICS:
                    met_obj = met(mod_name, j, self)
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
        self.guess_probability_dim(data_handler=data_handler)
        for met in metric_set[id(trainer.model)]:
            met.include(data_handler=data_handler)
    
    def guess_probability_dim(self, data_handler):
        """
        Assumes dimension is the first single dimension not shared by both model
        output and target, determined by size of dimension, or simply that 
        batch is dimension 0 followed by the label dimension, 1.
        """
        if (self.target_label_probability_dim is True
            or self.model_label_probability_dim is True):
            out_size = data_handler.output.size()
            tgt_size = data_handler.output.size()
            if len(out_size) == len(tgt_size):
                # Cannot surmise which dimension represents prob_dim, assume 1
                if self.target_label_probability_dim is True:
                    self.target_label_probability_dim = 1
                if self.model_label_probability_dim is True:
                    self.model_label_probability_dim = 1
            else:
                # First mismatched dimension supposed to represent prob_dim for
                # the longer tensor.
                rpt = [-1,0]
                prob_dim = -1
                for i, (o_dim, t_dim) in enumerate(zip(out_size, tgt_size)):
                    if o_dim != t_dim:
                        prob_dim = i
                        break
                    if rpt[0] == o_dim:
                        rpt[1] += 1
                    else:
                        rpt = [o_dim, 0]
                if prob_dim == -1:
                    # None mismatched, prob_dim assumed to be the next dim in
                    # the longer tensor.
                    if len(out_size) > len(tgt_size):
                        if self.model_label_probability_dim is True:
                            self.model_label_probability_dim = len(out_size)
                            self.target_label_probability_dim = False
                    else:
                        if self.target_label_probability_dim is True:
                            self.target_label_probability_dim = len(tgt_size)
                            self.model_label_probability_dim = False
                else:
                    # prob_dim assumed to be the dim in the longer tensor.
                    prob_dim -= rpt[1]
                    if len(out_size) > len(tgt_size):
                        if self.model_label_probability_dim is True:
                            self.model_label_probability_dim = prob_dim
                            self.target_label_probability_dim = False
                    else:
                        if self.target_label_probability_dim is True:
                            self.target_label_probability_dim = prob_dim
                            self.model_label_probability_dim = False