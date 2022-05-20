from typing import Any, List
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

from pytorch_lightning import LightningModule
from src.metrics.metrics import Countspecific, ClassificationMetrics
import src.models.modules.celltrack_model as celltrack_model


class CellTrackLitModel(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    """

    def __init__(
        self,
        sample,
        weight_loss,
        directed,
        model_params,
        separate_models,
        loss_weights,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.separate_models = separate_models
        if self.separate_models:
            model_attr = getattr(celltrack_model, model_params.target)
            self.model = model_attr(**model_params.kwargs)
        else:
            assert False, "Variable separate_models should be set to True!"
        self.sample = sample
        self.weight_loss = weight_loss

        # loss function
        if self.hparams.one_hot_label:
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_weights))
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        
        self.trClassMetric, self.valClassMetric, self.testClassMetric = \
                ClassificationMetrics(), ClassificationMetrics(), ClassificationMetrics()
        
        self.train_PredCount = Countspecific()
        self.val_PredCount = Countspecific()
        self.test_PredCount = Countspecific()

        self.train_TarCount, self.val_TarCount, self.test_TarCount = Countspecific(), Countspecific(), Countspecific()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }

    def forward(self, x, edge_index, edge_feat):
        return self.model(x, edge_index, edge_feat)

    def _compute_loss(self, outputs, edge_labels):
        edge_sum = edge_labels.sum()
        weight = (edge_labels.shape[0] - edge_sum) / edge_sum if edge_sum else 0.0
        loss = F.binary_cross_entropy_with_logits(outputs.view(-1),
                                                  edge_labels.view(-1),
                                                  pos_weight=weight).to(self.device)
        return loss

    def step(self, batch):
        if self.separate_models:
            x, x_2, edge_index, batch_ind, edge_label, edge_feat = batch.x, batch.x_2, batch.edge_index, batch.batch, batch.edge_label, batch.edge_feat
            y_hat = self.forward((x, x_2), edge_index, edge_feat.float())
        else:
            x, edge_index, batch_ind, edge_label, edge_feat = batch.x, batch.edge_index, batch.batch, batch.edge_label, batch.edge_feat
            y_hat = self.forward(x, edge_index, edge_feat.float())

        loss = self._compute_loss(y_hat, edge_label)
        preds = (y_hat >= 0.5).type(torch.int16)
        edge_label = edge_label.type(torch.int16)
        return loss, preds, edge_label

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        preds_sum, tar_sum = self.train_PredCount(preds), self.train_TarCount(targets)
        acc, prec, rec = self.trClassMetric(preds, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/prec", prec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/rec", rec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/preds_sum", preds_sum, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/targets_sum", tar_sum, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(self.device)
        self.logger[0].experiment.add_scalars('loss_epoch', {'train': avg_loss}, global_step=self.current_epoch)

        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

        acc, prec, rec = self.trClassMetric.compute()
        self.logger[0].experiment.add_scalar('train/acc_epoch', acc, self.current_epoch)
        self.logger[0].experiment.add_scalar('train/prec_epoch', prec, self.current_epoch)
        self.logger[0].experiment.add_scalar('train/recall_epoch', rec, self.current_epoch)
        self.logger[0].experiment.add_scalar('train/preds_sum_epoch', self.train_PredCount.compute(), self.current_epoch)
        self.logger[0].experiment.add_scalar('train/tar_sum_epoch', self.train_TarCount.compute(), self.current_epoch)

        self.trClassMetric.reset()
        self.train_PredCount.reset()
        self.train_TarCount.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        preds_sum, tar_sum = self.val_PredCount(preds), self.val_TarCount(targets)
        acc, prec, rec = self.valClassMetric(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/prec", prec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/rec", rec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/preds_sum", preds_sum, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/targets_sum", tar_sum, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(self.device)
        self.logger[0].experiment.add_scalars('loss_epoch', {'val': avg_loss}, global_step=self.current_epoch)

        # log best so far val acc and val loss
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)
        acc, prec, rec = self.valClassMetric.compute()
        self.logger[0].experiment.add_scalar('val/acc_epoch',       acc, self.current_epoch)
        self.logger[0].experiment.add_scalar('val/prec_epoch',      prec, self.current_epoch)
        self.logger[0].experiment.add_scalar('val/recall_epoch',    rec, self.current_epoch)
        self.logger[0].experiment.add_scalar('val/preds_sum_epoch', self.val_PredCount.compute(), self.current_epoch)
        self.logger[0].experiment.add_scalar('val/tar_sum_epoch',   self.val_TarCount.compute(), self.current_epoch)

        self.valClassMetric.reset()
        self.val_PredCount.reset()
        self.val_TarCount.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        preds_sum, tar_sum = self.test_PredCount(preds), self.test_TarCount(targets)

        acc, prec, rec = self.testClassMetric(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/prec", prec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/rec", rec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/preds_sum", preds_sum, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/targets_sum", tar_sum, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        acc, prec, rec = self.testClassMetric.compute()
        TP = self.testClassMetric.TP
        FP = self.testClassMetric.FP
        TN = self.testClassMetric.TN
        FN = self.testClassMetric.FN
        self.log("test/TP_epoch", TP, prog_bar=True)
        self.log("test/FP_epoch", FP, prog_bar=True)
        self.log("test/TN_epoch", TN, prog_bar=True)
        self.log("test/FN_epoch", FN, prog_bar=True)
        self.log("test/acc_epoch", acc, prog_bar=True)
        self.log("test/prec_epoch", prec, prog_bar=True)
        self.log("test/rec_epoch", rec, prog_bar=True)
        self.log("test/preds_sum_epoch", self.test_PredCount.compute(), prog_bar=True)
        self.log("test/targets_sum_epoch", self.test_TarCount.compute(), prog_bar=True)

        self.testClassMetric.reset()

        self.test_PredCount.reset()
        self.test_TarCount.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        """
        optim_class = getattr(optim, self.hparams.optim_module.target)
        optimizer = optim_class(params=self.model.parameters(), **self.hparams.optim_module.kwargs)

        if self.hparams.lr_sch_module.target is not None:
            lr_sch_class = getattr(lr_scheduler, self.hparams.lr_sch_module.target)
            lr_sch = lr_sch_class(optimizer=optimizer, **self.hparams.lr_sch_module.kwargs)
            assert self.hparams.lr_sch_module.monitor is not None, "Set monitor metric to track by..."
            return {"optimizer": optimizer, "lr_scheduler": lr_sch, "monitor": self.hparams.lr_sch_module.monitor}

        return optimizer
