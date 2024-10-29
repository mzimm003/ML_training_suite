from ml_training_suite.base import ML_Element
from ml_training_suite.registry import Registry
from ml_training_suite.datasets import DataHandler

import torch

import numpy as np

class Metric(ML_Element, register=False):
    registry = Registry()
    def __init__(self, model_name, fold_num, callback) -> None:
        self.callback = callback
        self.model_name = model_name
        self.fold_num = fold_num
        self.numerator = 0.
        self.denominator = 0.
    def include(self, num, den, data_handler:DataHandler=None):
        self.numerator += num
        self.denominator += den
    def get_result(self):
        return np.float64(self.numerator)/self.denominator
    def process_output_target(self, output, target):
        if not self.callback.model_label_probability_dim is False:
            output = output.max(self.callback.model_label_probability_dim).indices
        if not self.callback.target_label_probability_dim is False:
            target = target.max(self.callback.target_label_probability_dim).indices
        return output, target
    def __str__(self):
        return self.__class__.__name__

class Rate(Metric, register=False):
    def include(self, num, den=1):
        super().include(num, den)

class Loss(Rate):
    def include(self, data_handler:DataHandler):
        super().include(data_handler.loss.item())

class LearningRate(Rate):
    def include(self, data_handler:DataHandler):
        super().include(data_handler.get_last_lr())

class pAUC(Rate):
    def __init__(self, model_name, fold_num, callback, class_idx=-1, p=0.8) -> None:
        """
        Args:
            class_idx: Index within the class distribution of the class to be measured.
        """
        super().__init__(model_name, fold_num, callback)
        self.p = p
        self.confidences = []
        self.targets = []
        self.class_idx = class_idx
    
    def include(self, data_handler: DataHandler):
        confidences = self.get_confidences(data_handler=data_handler)
        self.confidences.append(confidences.detach().cpu())
        self.targets.append(data_handler.target.detach().cpu())
    
    def get_result(self):
        positive_class_start_idx, tgts_sorted = self.get_classifications_and_tgts_sorted(
            confidences=torch.cat(self.confidences),
            targets=torch.cat(self.targets)
        )
        tp = torch.tensor([(tgts_sorted[i:]==True).sum() for i in positive_class_start_idx])
        tpr = tp/(tgts_sorted==True).sum()
        fp = torch.tensor([(tgts_sorted[i:]==False).sum() for i in positive_class_start_idx])
        fpr = fp/(tgts_sorted==False).sum()
        rect_heights = (tpr-self.p).clip(0)
        rect_widths = torch.diff(fpr, append=torch.tensor([0], device=fpr.device)).abs()
        pauc = (rect_heights*rect_widths).sum()
        return pauc
    
    def get_confidences(self, data_handler: DataHandler):
        confidences = data_handler.output.softmax(self.callback.model_label_probability_dim)
        confidences = confidences[:,self.class_idx]
        return confidences

    def get_classifications_and_tgts_sorted(
            self,
            data_handler: DataHandler=None,
            confidences=None,
            targets=None):
        assert not data_handler is None or not (confidences is None or targets is None)
        confidences = (confidences if
                        not confidences is None else
                        self.get_confidences(data_handler=data_handler))
        con_sorted, con_sort_indxs = confidences.sort()
        targets = targets if not targets is None else data_handler.target
        tgts_sorted = targets[con_sort_indxs]
        _, counts = torch.unique_consecutive(con_sorted, return_counts=True)
        indices = torch.cumsum(counts, dim=0) - counts
        positive_class_start_idx = torch.repeat_interleave(indices, counts)
        return positive_class_start_idx, tgts_sorted    

class Ratio(Metric, register=False):
    pass

class Accuracy(Ratio):
    def include(self, data_handler: DataHandler):
        output, target = self.process_output_target(
            data_handler.output,
            data_handler.target)
        num_matches = (output.int() == target.int()).sum()
        num_total = target.numel()
        super().include(
            num_matches.item(),
            num_total)


class Precision(Ratio):
    def include(self, data_handler: DataHandler):
        output, target = self.process_output_target(
            data_handler.output,
            data_handler.target)
        num_true_positives=(
            target.int()[output.int() == 1].sum())
        num_positive_inferences = (output.int() == 1).sum()
        super().include(
            num_true_positives.item(),
            num_positive_inferences.item()
            )

class Recall(Ratio):
    def include(self, data_handler: DataHandler):
        output, target = self.process_output_target(
            data_handler.output,
            data_handler.target)
        num_true_positives=(
            target.int()[output.int() == 1].sum())
        num_positive_targets = (target.int() == 1).sum()
        super().include(
            num_true_positives.item(),
            num_positive_targets.item()
            )
