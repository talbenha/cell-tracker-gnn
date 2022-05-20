from torchmetrics import Metric
import torch


class Countspecific(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, one_zero_array: torch.Tensor):
        assert not torch.any(torch.bitwise_and(one_zero_array != 0, one_zero_array != 1)), \
            "input consists of values that are different from 1 and 0"
        self.total += one_zero_array.sum()

    def compute(self):
        return self.total.float()


class ClassificationMetrics(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("TP", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FP", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("TN", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FN", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert not torch.any(torch.bitwise_and(preds != 0, preds != 1)), \
            "input consists of values that are different from 1 and 0"

        assert not torch.any(torch.bitwise_and(target != 0, target != 1)), \
            "input consists of values that are different from 1 and 0"

        with torch.no_grad():
            self.TP += ((target == 1) & (preds == 1)).sum().float()
            self.FP += ((target == 0) & (preds == 1)).sum().float()
            self.TN += ((target == 0) & (preds == 0)).sum().float()
            self.FN += ((target == 1) & (preds == 0)).sum().float()

    def compute(self):
        TP = self.TP
        FP = self.FP
        TN = self.TN
        FN = self.FN

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        recall = TP / (TP + FN) if TP + FN > 0 else TP
        precision = TP / (TP + FP) if TP + FP > 0 else TP
        return accuracy, recall, precision


class ClassificationMetrics_Inference(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("TP", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FP", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("TN", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FN", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert not torch.any(torch.bitwise_and(preds != 0, preds != 1)), \
            "input consists of values that are different from 1 and 0"

        assert not torch.any(torch.bitwise_and(target != 0, target != 1)), \
            "input consists of values that are different from 1 and 0"

        with torch.no_grad():
            self.TP += ((target == 1) & (preds == 1)).sum().float()
            self.FP += ((target == 0) & (preds == 1)).sum().float()
            self.TN += ((target == 0) & (preds == 0)).sum().float()
            self.FN += ((target == 1) & (preds == 0)).sum().float()

    def compute(self):
        TP = self.TP
        FP = self.FP
        TN = self.TN
        FN = self.FN

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        recall = TP / (TP + FN) if TP + FN > 0 else TP
        precision = TP / (TP + FP) if TP + FP > 0 else TP

        return accuracy, recall, precision, TP, FP, TN, FN
