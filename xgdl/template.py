from baselines import *

class DataUtil:
    def build():
        pass
    def get_train_dataloader():
        pass
    def get_val_dataloader():
        pass
    def get_test_dataloader():
        pass

from enum import Enum
class ModelUtil(Enum):
    model1 = ASAP
    model2 = LRIBern
    def build(self, args):
        cls_name = args.model_name
        self.model = self.cls_name.value(args)
        return self.model
    
    @staticmethod
    def save(model, metric, path):
        import torch
        if metric.best_perf.value > metric.compute():
            torch.save(model.state_dict(), path)


class OptimizerUtil:
    def build(self, args):
        return None

from collections import namedtuple

class Trainer:
    BestPerf = namedtuple("BestPerf", ['epoch', 'value'])
    def __init__(self, args) -> None:
        self.data = DataUtil.build(args)
        self.model = ModelUtil.build(args)
        self.optimizer = OptimizerUtil.build(args)
        self.metric = None
        self.loss_fn = self.optimizer.loss_fn

    def evaluate(self, dataset, model, optimizer):
        self.metric.reset()
        for batch in range(self.data):
            data, targets = batch
            self._eval_batch(data, targets)
        epoch_ret = self.metric.compute()
        return epoch_ret

    def train(self, val_dataset, model, trial=None, ckpt=None):
        info = print
        metric = self.metric
        for epoch in range(10):
            metric.reset()    
            for batch in range(self.data):
                data, target = batch
                self._train_batch(data, target) 
            epoch_ret = metric.compute()

            # compute best performance for epoch selection
            metric.best_perf = Trainer.BestPerf(epoch, max(metric.best_perf, epoch_ret))

            # report result
            info(epoch_ret, metric.best_perf)

            # save model
            ModelUtil.save(self.model, metric, path='')
        info("All interations done")

    def predict(self, **kwargs):
        self.evaluate(**kwargs)

    def _eval_batch(self, data, target):
        #shared by predict and evaluate
        preds = self.model(data) 
        ret = self.metric.update(preds, target)
        return ret

    def _train_batch(self, data, targets):
        self.optimizer.zero_grad() # reset grad
        preds = self.model(data)
        ret = self.metric.update(preds, targets) # update metric
        loss = self.loss_fn(preds, targets)
        loss.backward() # update grad
        self.optimizer.step() # compute grad