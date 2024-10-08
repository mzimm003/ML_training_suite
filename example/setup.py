
from ml_training_suite.datasets import Dataset, DataHandlerGenerator
from ml_training_suite.training import Criterion, Trainer
from ml_training_suite.models import Model
from ml_training_suite.models.elements import Activation

import torch
from torch import nn

import pandas as pd

import numpy as np

import h5py
from PIL import Image
import io


from typing import (
    Union,
    Any,
    List
)
from typing_extensions import override
from pathlib import Path

class ISIC(Trainer):
    @override
    def map_data_handler(self, pipeline):
        return DataHandlerGenerator(
            img=SimpleCustomBatch.IMG,
            fet=SimpleCustomBatch.FET,
            target=SimpleCustomBatch.TGT,
            pipeline=pipeline if pipeline else [])

class SimpleCustomBatch:
    IMG="img"
    FET="fet"
    TGT="tgt"
    def __init__(self, data):
        transposed_data = list(zip(*data))
        transposed_inps = list(zip(*transposed_data[0]))
        self.img = torch.stack(transposed_inps[0], 0)
        self.fet = torch.stack(transposed_inps[1], 0)
        self.tgt = None
        if isinstance(transposed_data[1][0], str):
            self.tgt = transposed_data[1]
        else:
            self.tgt = torch.stack(transposed_data[1], 0).long()

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.img = self.img.pin_memory()
        self.fet = self.fet.pin_memory()
        if not isinstance(self.tgt, tuple):
            self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

class SkinLesions(Dataset):
    collate_wrapper = collate_wrapper
    def __init__(
            self,
            annotations_file:Union[str, Path],
            img_file:Union[str, Path] = None,
            img_dir:Union[str, Path] = None,
            img_transform:nn.Module = None,
            annotation_transform:nn.Module = None,
            annotations_only:bool = False,
            label_desc:str = None,
            ret_id_as_label:bool = False):
        metadata = pd.read_csv(annotations_file, low_memory=False)
        
        self.label_desc = label_desc
        self.annotations_only = annotations_only
        self.img_file = None
        self.img_dir = None
        self.img_listing = None
        self.ret_id_as_label = ret_id_as_label

        if not self.annotations_only:
            self.img_file = img_file
            self.img_dir = img_dir
            self.img_listing = metadata.loc[:, "isic_id"]

        self.img_transform = img_transform
        self.annotation_transform = annotation_transform
        if self.annotation_transform:
            metadata = self.annotation_transform(metadata)
        self.annotations = None
        self.labels = None
        if self.label_desc:
            self.annotations = torch.tensor(
                metadata.drop(self.label_desc, axis=1).values,
                dtype=torch.float32)
            self.labels = torch.tensor(
                metadata[self.label_desc],
                dtype=torch.float32)
        else:
            self.annotations = torch.tensor(metadata.values, dtype=torch.float32)
            self.labels = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = None
        label = torch.tensor(-1, dtype=torch.float32)
        annotations = self.annotations[idx]
        if self.label_desc:
            label = self.labels[idx]

        if self.annotations_only:
            data = annotations
        else:
            image_data = h5py.File(self.img_file, "r")
            listing = self.img_listing[idx]
            if self.ret_id_as_label:
                label = listing
            image = np.array(image_data[listing])
            image = np.array(Image.open(io.BytesIO(image)),dtype=np.uint8)
            if self.img_transform:
                image = self.img_transform(image)
            data = (
                torch.tensor(image, dtype=torch.float32),
                annotations
                )

        return data, label

class SkinLesionsSmall(SkinLesions):
    def __init__(
            self,
            annotations_file: str | Path,
            img_file: str | Path = None,
            img_dir: str | Path = None,
            img_transform: nn.Module = None,
            annotation_transform: nn.Module = None,
            annotations_only: bool = False,
            label_desc: str = None,
            ret_id_as_label: bool = False,
            size:int = 2048):
        super().__init__(annotations_file, img_file, img_dir, img_transform, annotation_transform, annotations_only, label_desc, ret_id_as_label)
        rng = np.random.default_rng()
        if not self.labels is None:
            unq_lbls = self.labels.unique()
            lbl_masks = self.labels==unq_lbls[:,None]
            not_lbl_counts = (self.labels!=unq_lbls[:,None]).float().sum(-1)[:,None]
            weights = (lbl_masks*not_lbl_counts).sum(0).double()
            weights = weights/weights.sum()
            mask = rng.choice(len(self.labels), size, replace=False, p=weights)
            self.labels = self.labels[mask]
            self.annotations = self.annotations[mask]
        else:
            mask = rng.choice(len(self.annotations), size, replace=False)
            self.annotations = self.annotations[mask]

class Classifier(Model):
    def __init__(
            self,
            embedding_dim:int = 64,
            img_height:int = 125,
            img_width:int = 125,
            patch_size:int = 5,
            nhead:int = 8,
            layers:int = 4,
            dim_feedforward:int = 1024,
            norm_first:bool = False,
            activation:Union[str, Activation, nn.Module] = None,
            activation_kwargs:dict[str, Any] = None,
            *args,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = Activation.initialize(
            activation, activation_kwargs if activation_kwargs else {})
        self.feature_embedding = nn.Sequential(
            nn.LazyLinear(embedding_dim),
            self.activation,
            nn.Linear(embedding_dim, embedding_dim))
        
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        assert (self.img_height % self.patch_size == 0 and
                self.img_width % self.patch_size == 0)
        self.positional_embedding_x = nn.Embedding(
            self.img_width//self.patch_size, embedding_dim)
        self.positional_embedding_y = nn.Embedding(
            self.img_height//self.patch_size, embedding_dim)
        self.img_reshape_x_mask = (
            torch.arange(self.patch_size).repeat(self.patch_size)+
            (torch.arange(self.img_width//self.patch_size)*self.patch_size)[:,None]
        ).repeat(self.img_height//self.patch_size, 1).long()
        self.img_reshape_y_mask = (
            torch.arange(self.patch_size).repeat_interleave(self.patch_size)+
            (torch.arange(self.img_height//self.patch_size)*self.patch_size)[:,None]
            ).repeat_interleave(self.img_width//self.patch_size, 0).long()
        self.reshape_img = lambda x: x[..., self.img_reshape_y_mask, self.img_reshape_x_mask, :].flatten(-2)
        self.img_patch_embedding = nn.LazyLinear(embedding_dim)
        self.img_flatten = nn.Flatten(-3,-2)
        self.transformer = nn.Transformer(
            d_model = embedding_dim,
            nhead = nhead,
            num_encoder_layers = layers,
            num_decoder_layers = layers,
            dim_feedforward = dim_feedforward,
            activation = self.activation,
            batch_first = True,
            norm_first = norm_first,
        )
        self.is_malignant = nn.Sequential(
            nn.Linear(embedding_dim, 2),
            self.activation,
            nn.Flatten(-2),
            nn.LazyLinear(2)
        )
    
    def forward(self, img:torch.Tensor, fet:torch.Tensor):
        param_ref = next(self.transformer.parameters())
        fet = fet.to(dtype=param_ref.dtype, device=param_ref.device)
        img = img.to(dtype=param_ref.dtype, device=param_ref.device)

        fet = self.feature_embedding(fet[:,None]*torch.eye(fet.size(-1), device=param_ref.device))

        width = img.size(-3)
        height = img.size(-2)
        x_emb = self.positional_embedding_x(
            torch.arange(width//self.patch_size, device=param_ref.device))
        y_emb = self.positional_embedding_y(
            torch.arange(height//self.patch_size, device=param_ref.device))
        pos_emb = (
            torch.tile(x_emb[None,:], (height//self.patch_size,1,1)) +
            torch.tile(y_emb[:,None], (1,width//self.patch_size,1))
            )
        pos_emb = self.img_flatten(pos_emb)
        img = self.reshape_img(img)
        im_emb = self.img_patch_embedding(img)
        enc_mem = self.transformer.encoder(im_emb + pos_emb)
        aux_logits = []
        a_l = fet
        for l in self.transformer.decoder.layers:
            a_l = l(a_l, enc_mem)
            aux_logits.append(self.is_malignant(a_l))
        a_l = self.transformer.decoder.norm(a_l)
        logits = self.is_malignant(a_l)
        return logits, *aux_logits

    def name(self):
        return "Classifier1.6"
    
class ClassifierLoss(nn.CrossEntropyLoss, Criterion):
    def __init__(self, main_weight=0.5, aux_weight=0.5,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_weight = main_weight
        self.aux_weight = aux_weight

    def forward(
            self,
            input: torch.Tensor,
            aux_inputs: List[torch.Tensor],
            target: torch.Tensor) -> torch.Tensor:
        main_loss = super().forward(input, target)
        aux_loss = 0
        for aux_inp in aux_inputs:
            aux_loss += super().forward(aux_inp, target)
        return main_loss*self.main_weight + aux_loss*self.aux_weight