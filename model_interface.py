# Copyright 2024 Haowen Yu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as pl

from loss.loss_funcs import cross_entropy_loss
from utils.metrics.sequence_classification import save_confusion_matrix, top1_accuracy, \
    top5_accuracy, time_to_correct_decision, early_classification_accuracy, auc_over_accuracy_curve

from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import itertools
import os


class ModelInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.__load_model()
        # self.model = self.model.to(torch.device('cuda'))
        self.loss_function = self.__configure_loss()
        self.train_epoch_correct = 0
        self.train_epoch_total = 0
        self.val_epoch_correct = 0
        self.val_epoch_total = 0
        self.log_dir_full = kwargs['log_dir_full']
        self.train_epoch_outputs = []
        self.val_epoch_outputs = []

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_end(self):
        top1_acc = top1_accuracy(self.train_epoch_outputs)
        self.log('train_acc_top1', top1_acc, on_step=False, on_epoch=True, prog_bar=True)

        top5_acc = top5_accuracy(self.train_epoch_outputs)
        self.log('train_acc_top5', top5_acc, on_step=False, on_epoch=True, prog_bar=True)

        train_TTCD, _ = time_to_correct_decision(self.train_epoch_outputs, window_size=5)
        self.log('train_TTCD', train_TTCD, on_step=False, on_epoch=True, prog_bar=True)

        train_ECA = early_classification_accuracy(self.train_epoch_outputs)
        self.log('train_ECA@20%', train_ECA[20], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_ECA@30%', train_ECA[30], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_ECA@40%', train_ECA[40], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_ECA@50%', train_ECA[50], on_step=False, on_epoch=True, prog_bar=True)

        train_AUCOAC = auc_over_accuracy_curve(acc_dict=train_ECA)
        self.log('train_AUCOAC', train_AUCOAC, on_step=False, on_epoch=True, prog_bar=True)

        class_names = [str(i) for i in range(self.hparams.num_classes)]
        save_confusion_matrix(self.train_epoch_outputs, class_names=class_names, 
                              save_path=os.path.join(self.log_dir_full, "train_confusion_matrix.png"), normalize=False)

        del self.train_epoch_outputs
        self.train_epoch_outputs = []

    def on_validation_epoch_end(self):
        top1_acc = top1_accuracy(self.val_epoch_outputs)
        self.log('val_acc_top1', top1_acc, on_step=False, on_epoch=True, prog_bar=True)

        top5_acc = top5_accuracy(self.val_epoch_outputs)
        self.log('val_acc_top5', top5_acc, on_step=False, on_epoch=True, prog_bar=True)

        val_TTCD, _ = time_to_correct_decision(self.val_epoch_outputs, window_size=5)
        self.log('train_TTCD', val_TTCD, on_step=False, on_epoch=True, prog_bar=True)

        val_ECA = early_classification_accuracy(self.val_epoch_outputs)
        self.log('val_ECA@20%', val_ECA[20], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ECA@30%', val_ECA[30], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ECA@40%', val_ECA[40], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ECA@50%', val_ECA[50], on_step=False, on_epoch=True, prog_bar=True)

        val_AUCOAC = auc_over_accuracy_curve(acc_dict=val_ECA)
        self.log('val_AUCOAC', val_AUCOAC, on_step=False, on_epoch=True, prog_bar=True)

        class_names = [str(i) for i in range(self.hparams.num_classes)]
        save_confusion_matrix(self.val_epoch_outputs, class_names=class_names, 
                              save_path=os.path.join(self.log_dir_full, "val_confusion_matrix.png"), normalize=False)

        del self.val_epoch_outputs
        self.val_epoch_outputs = []

    # Caution: self.model.train() is invoked
    def training_step(self, batch, batch_idx):
        train_input, valid_len, train_labels = batch   # train_labels: [B] long, class indices
        train_out = self(train_input)                  # [B, T, C]

        B, T, C = train_out.shape
        last_idx = valid_len - 1
        train_out_last = train_out[torch.arange(B), last_idx]   # [B, C]

        # if using CrossEntropyLoss, labels should be [B] integer indices
        train_loss = self.loss_function(train_out_last, train_labels, 'train')

        self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True)

        if self.hparams.model_class_name == 'ssm':
            self.model.reset_state(B)

        topk = 5
        train_topk_preds = train_out.topk(k=topk, dim=-1).indices.cpu().numpy()  # shape (B, T, topk)

        train_batch_output = {
            "loss": train_loss,
            "preds": train_topk_preds,
            "labels": train_labels.cpu().numpy(),
            "valid_len": valid_len.cpu().numpy(),
        }

        self.train_epoch_outputs.append(train_batch_output)

        return train_batch_output

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def validation_step(self, batch, batch_idx):
        val_input, valid_len, val_labels = batch
        val_out = self(val_input)
    
        B, T, C = val_out.shape
        last_idx = valid_len - 1   # valid_len counts length, so subtract 1 for index
        # Gather the logits at last valid index
        val_out_last = val_out[torch.arange(B), last_idx]   # [B, num_classes]
        val_loss = self.loss_function(val_out_last, val_labels, 'validation')

        self.log('val_loss', val_loss, on_step=True, on_epoch=False, prog_bar=True)

        if self.hparams.model_class_name == 'ssm':
            self.model.reset_state(B)

        topk = 5
        val_topk_preds = val_out.topk(k=topk, dim=-1).indices.cpu().numpy()  # shape (B, T, topk)

        val_batch_output = {
            "loss": val_loss,
            "preds": val_topk_preds,
            "labels": val_labels.cpu().numpy(),
            "valid_len": valid_len.cpu().numpy(),
        }

        self.val_epoch_outputs.append(val_batch_output)

        return val_batch_output

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def test_step(self, batch, batch_idx):
        # Same with validation_step except for the log verbose info
        test_input, valid_len, test_labels = batch
        test_out = self(test_input)
        test_loss = self.loss_function(test_out, test_labels, 'test')

        test_preds = test_out.argmax(axis=1).detach()
        correct_num = torch.sum(test_preds == test_labels).cpu().item()

        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return test_loss

    # When there are multiple optimizers, modify this function to fit in your needs
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay)
        )

        # No learning rate scheduler, just return the optimizer
        if self.hparams.lr_scheduler is None:
            return [optimizer]

        # Return tuple of optimizer and learning rate scheduler
        if self.hparams.lr_scheduler == 'step':
            scheduler = lrs.StepLR(
                optimizer,
                step_size=self.hparams.lr_decay_epochs,
                gamma=self.hparams.lr_decay_rate
            )
        elif self.hparams.lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.lr_decay_epochs,
                eta_min=self.hparams.lr_decay_min_lr
            )
        else:
            raise ValueError('Invalid lr_scheduler type!')
        return [optimizer], [scheduler]

    def __configure_loss(self):
        def loss_func(inputs, labels, stage):
            return cross_entropy_loss(inputs, labels)

        return loss_func

    def __load_model(self):
        name = self.hparams.model_class_name
        # Attempt to import the `CamelCase` class name from the `snake_case.py` module. The module should be placed
        # within the same folder as model_interface.py. Always name your model file name as `snake_case.py` and
        # model class name as corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            model_class = getattr(importlib.import_module('model.' + name, package=__package__), camel_name)
        except Exception:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        model = self.__instantiate(model_class)
        if self.hparams.use_compile:
            torch.compile(model)
        return model

    def __instantiate(self, model_class, **other_args):
        # Instantiate a model using the imported class name and parameters from self.hparams dictionary.
        # You can also input any args to overwrite the corresponding value in self.hparams.
        target_args = inspect.getfullargspec(model_class.__init__).args[1:]
        this_args = self.hparams.keys()
        merged_args = {}
        # Only assign arguments that are required in the user-defined torch.nn.Module subclass by their name.
        # You need to define the required arguments in main function.
        for arg in target_args:
            if arg in this_args:
                merged_args[arg] = getattr(self.hparams, arg)

        merged_args.update(other_args)
        return model_class(**merged_args)
