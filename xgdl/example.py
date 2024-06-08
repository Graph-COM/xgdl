from baselines import LRIBern
from eval import ModelEval, DataEval 


class InherentExplainer:
    model = LRIBern
    config = None
    type = 'inherent'
    def __init__(self) -> None:
        pass

class PosthocExplainer:
    model = None
    config = None


def prepare_trainer():
    pass


data_loader = None

class Trainer:
    pass

trainer = Trainer()


trainer.warmup()

trainer.train()

trainer.load()

trainer.explain(data_loader)




