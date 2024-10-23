from ml_training_suite.base import ML_Element
from ml_training_suite.registry import Registry
from ml_training_suite.datasets import DataHandler

import torch

import numpy as np

class Metric(ML_Element, register=False):
    registry = Registry()
    def __init__(self, model_name, fold_num) -> None:
        self.model_name = model_name
        self.fold_num = fold_num
        self.numerator = 0.
        self.denominator = 0.
    def include(self, num, den, data_handler:DataHandler=None):
        self.numerator += num
        self.denominator += den
    def get_result(self):
        return np.float64(self.numerator)/self.denominator
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
    def __init__(self, model_name, fold_num, p=0.8) -> None:
        super().__init__(model_name, fold_num)
        self.p = p
        self.confidences = []
        self.targets = []
    
    def include(self, data_handler: DataHandler):
        mal_confidences = get_mal_confidences(data_handler=data_handler)
        self.confidences.append(mal_confidences.detach().cpu())
        self.targets.append(data_handler.target.detach().cpu())
    
    def get_result(self):
        positive_class_start_idx, tgts_sorted = get_classifications_and_tgts_sorted(
            mal_confidences=torch.cat(self.confidences),
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

class Ratio(Metric, register=False):
    pass

class Accuracy(Ratio):
    def include(self, data_handler: DataHandler):
        num_matches = (
            data_handler.output_label.int() == data_handler.target.int()).sum()
        num_total = data_handler.target.numel()
        super().include(
            num_matches.item(),
            num_total)

def get_mal_confidences(data_handler: DataHandler):
    confidences = data_handler.output.softmax(-1)
    mal_confidences = confidences[:,-1]
    return mal_confidences

def get_classifications_and_tgts_sorted(
        data_handler: DataHandler=None,
        mal_confidences=None,
        targets=None):
    assert not data_handler is None or not (mal_confidences is None or targets is None)
    mal_confidences = (mal_confidences if
                       not mal_confidences is None else
                       get_mal_confidences(data_handler=data_handler))
    mal_con_sorted, mal_con_sort_indxs = mal_confidences.sort()
    targets = targets if not targets is None else data_handler.target
    tgts_sorted = targets[mal_con_sort_indxs]
    _, counts = torch.unique_consecutive(mal_con_sorted, return_counts=True)
    indices = torch.cumsum(counts, dim=0) - counts
    positive_class_start_idx = torch.repeat_interleave(indices, counts)
    return positive_class_start_idx, tgts_sorted    

class Precision(Ratio):
    def include(self, data_handler: DataHandler):
        num_true_positives=(
            data_handler.target.int()[data_handler.output_label.int() == 1].sum())
        num_positive_inferences = (data_handler.output_label.int() == 1).sum()
        super().include(
            num_true_positives.item(),
            num_positive_inferences.item()
            )

class Recall(Ratio):
    def include(self, data_handler: DataHandler):
        num_true_positives=(
            data_handler.target.int()[data_handler.output_label.int() == 1].sum())
        num_positive_targets = (data_handler.target.int() == 1).sum()
        super().include(
            num_true_positives.item(),
            num_positive_targets.item()
            )