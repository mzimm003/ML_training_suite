from ml_training_suite.training.base import ray_trainable_wrap as trainable_wrap
from ml_training_suite.training import (
    TrainingScript,
    Optimizer,
    LRScheduler,
    Criterion,
    Trainer
)
from ml_training_suite.preprocess.features import PPLabels
from ml_training_suite.preprocess.images import PPPicture
from ml_training_suite.models import (
    Model,
)
from ml_training_suite.models.elements import Activation
from ml_training_suite.datasets import Dataset

import ray
from ray import tune
from ray import air

import torch

import os
from pathlib import Path

import setup

debug = False

def main():
    if debug:
        num_trials = 1
        num_cpus = 1 if debug else os.cpu_count()
        num_gpus = 0 if debug else torch.cuda.device_count()
        BATCHSIZE=256
        EPOCHS=100
        cpu_per_trial = num_cpus//num_trials
        gpu_per_trial = num_gpus/num_trials
        annotations_file=Path("/home/user/datasets/isic-2024-challenge/train-metadata.csv").resolve()
        img_file=Path("/home/user/datasets/isic-2024-challenge/train-image.hdf5").resolve()
        img_dir=Path("/home/user/datasets/isic-2024-challenge/train-image").resolve()
        feature_reducer_paths=[
            "/home/user/Programming/Kaggle Competitions/ISIC2024/models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            None,
            ]
        save_path=Path("./models/classifier").resolve()
        config = dict(
                dataset=Dataset.SkinLesionsSmall,
                dataset_kwargs=dict(
                    annotations_file=annotations_file,
                    img_file=img_file,
                    img_dir=img_dir,
                    img_transform=PPPicture(
                        pad_mode='edge',
                        pass_larger_images=True,
                        crop=True,
                        random_brightness=True,
                        random_contrast=True,
                        random_flips=True),
                    annotation_transform=PPLabels(
                        exclusions=[
                            "isic_id",
                            "patient_id",
                            "attribution",
                            "copyright_license",
                            "lesion_id",
                            "iddx_full",
                            "iddx_1",
                            "iddx_2",
                            "iddx_3",
                            "iddx_4",
                            "iddx_5",
                            "mel_mitotic_index",
                            "mel_thick_mm",
                            "tbp_lv_dnn_lesion_confidence",
                            ],
                        fill_nan_selections=[
                            "age_approx",
                        ],
                        fill_nan_values=[-1, 0],
                        ),
                    label_desc='target',),
                trainer_class = Trainer.ISIC,
                pipelines=[
                    [('fet', Path(pth).resolve())] if pth else pth
                    for pth in feature_reducer_paths]*2,
                models=[
                    Model.Classifier,
                    Model.Classifier,
                    Model.Classifier,
                    Model.Classifier,],
                models_kwargs=dict(
                    activation=Activation.relu),
                optimizers=Optimizer.adam,
                optimizers_kwargs=dict(
                    lr=0.00005
                ),
                lr_schedulers=[
                    None,
                    None,
                    LRScheduler.CyclicLR,
                    LRScheduler.CyclicLR],
                lr_schedulers_kwargs=[
                    None,
                    None,
                    dict(
                            base_lr=0.0000001,
                            max_lr=0.000025,
                            step_size_up=(401059//BATCHSIZE)*3
                        ),
                    dict(
                            base_lr=0.0000001,
                            max_lr=0.000025,
                            step_size_up=(401059//BATCHSIZE)*3
                        ),],
                criterion=Criterion.ClassifierLoss,
                criterion_kwargs=dict(
                    weight=torch.tensor([393/401059, 400666/401059])
                ),
                save_path=save_path,
                balance_training_set=False,
                k_fold_splits=1,
                batch_size=BATCHSIZE,
                shuffle=True,
                num_workers=cpu_per_trial-1,
            )
        script = TrainingScript.SupervisedTraining(**config)
        script.setup()
        results = script.run()

        for i, (mod, data_samp) in enumerate(script.get_models_for_onnx_save()):
            script.save_model(mod, data_samp, i)
            script.save_results(mod, results)
    else:
        num_trials = 1
        num_cpus = 1 if debug else os.cpu_count()
        num_gpus = 0 if debug else torch.cuda.device_count()
        BATCHSIZE=256
        EPOCHS=100
        cpu_per_trial = num_cpus//num_trials
        gpu_per_trial = num_gpus/num_trials
        annotations_file=Path("/home/user/datasets/isic-2024-challenge/train-metadata.csv").resolve()
        img_file=Path("/home/user/datasets/isic-2024-challenge/train-image.hdf5").resolve()
        img_dir=Path("/home/user/datasets/isic-2024-challenge/train-image").resolve()
        feature_reducer_paths=[
            "/home/user/Programming/Kaggle Competitions/ISIC2024/models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            None,
            ]
        save_path=Path("./models/classifier").resolve()
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            local_mode=debug,
            storage="/opt/ray/results"
            )
        tuner = tune.Tuner(
            trainable_wrap(
                TrainingScript.SupervisedTraining,
                num_cpus=cpu_per_trial,
                num_gpus=gpu_per_trial),
            run_config=air.RunConfig(
                name="TransformerClassifier",
                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=2),
                stop={"training_iteration": EPOCHS}),
            param_space=dict(
                dataset=Dataset.SkinLesionsSmall,
                dataset_kwargs=dict(
                    annotations_file=annotations_file,
                    img_file=img_file,
                    img_dir=img_dir,
                    img_transform=PPPicture(
                        pad_mode='edge',
                        pass_larger_images=True,
                        crop=True,
                        random_brightness=True,
                        random_contrast=True,
                        random_flips=True),
                    annotation_transform=PPLabels(
                        exclusions=[
                            "isic_id",
                            "patient_id",
                            "attribution",
                            "copyright_license",
                            "lesion_id",
                            "iddx_full",
                            "iddx_1",
                            "iddx_2",
                            "iddx_3",
                            "iddx_4",
                            "iddx_5",
                            "mel_mitotic_index",
                            "mel_thick_mm",
                            "tbp_lv_dnn_lesion_confidence",
                            ],
                        fill_nan_selections=[
                            "age_approx",
                        ],
                        fill_nan_values=[-1, 0],
                        ),
                    label_desc='target',),
                trainer_class = Trainer.ISIC,
                pipelines=[
                    [('fet', Path(pth).resolve())] if pth else pth
                    for pth in feature_reducer_paths]*2,
                models=[
                    Model.Classifier,
                    Model.Classifier,
                    Model.Classifier,
                    Model.Classifier,],
                models_kwargs=dict(
                    activation=Activation.relu),
                optimizers=Optimizer.adam,
                optimizers_kwargs=dict(
                    lr=tune.grid_search([0.00005])
                ),
                lr_schedulers=[
                    None,
                    None,
                    LRScheduler.CyclicLR,
                    LRScheduler.CyclicLR],
                lr_schedulers_kwargs=[
                    None,
                    None,
                    dict(
                            base_lr=0.0000001,
                            max_lr=0.000025,
                            step_size_up=(401059//BATCHSIZE)*3
                        ),
                    dict(
                            base_lr=0.0000001,
                            max_lr=0.000025,
                            step_size_up=(401059//BATCHSIZE)*3
                        ),],
                criterion=Criterion.ClassifierLoss,
                criterion_kwargs=dict(
                    weight=torch.tensor([393/401059, 400666/401059])
                ),
                save_path=save_path,
                balance_training_set=False,
                k_fold_splits=1,
                batch_size=BATCHSIZE,
                shuffle=True,
                num_workers=cpu_per_trial-1,
            )
        )
        tuner.fit()
        ray.shutdown()
    print("Done")

if __name__ == "__main__":
    main()