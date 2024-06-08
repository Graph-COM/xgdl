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



import PosthocExplainer, InherentExplainer

model = None
data = None
data_loader = None

#! all exlainers needs the model has a method named "get_embedding" and input data has attributes "pos". Can be configed to more models??

explainerconfig = {
    "name": 'which kind',
    "input_attr": "x" # 'pos'
}

InherentExplainer(explainerconfig, model=model)
explainer = PosthocExplainer(explainerconfig, modle=model)
explainer.train(data_loader)

prediction = explainer.predict(data)

#! in explain() we need model.get_emb(data), model.get_pred_from_emb(emb) and self.extractor(emb)
interpretion = explainer.explain(data)

class Evaluator():
    def eval():
        pass


def evaluate_explainer():
    pass

fidel = evaluate_explainer(explainer, data, metric='Sensitive')
auc = explainer.eval(data, interpretion, metric='Decisive')
