# XGDL
XGDL (eXplainability for Geometric Deep Learning) is an explainability library for scientific tasks using geometric deep learning.
______________________________________________________________________
## Quick Tour
```python
from typing import Any
from xgdl.backbones import DGCNN 
from transformers import AutoModel
from xgdl import PosthocMethod, InherentModel, HEPDataset

model = None
data = None
data_loader = None


init_kwargs = {
    'clf': 1,
    'extractor': 1,
    'criterion': 1,
}

config = {
    'pred_loss_coef': 0.1,
    'info_loss_coef': 0.05,
    'temperature': 1.0,
    'final_r': 0.9,
    'decay_interval': 10,
    'decay_r': 0.01,
    'init_r': 0.5,
    'attn_constraint': True
}
#! all exlainers needs the model has a method named "get_embedding" and input data has attributes "pos". 
lri_model = InherentModel.from_name('lri_bern', init_kwargs)


explainerconfig = {
    "name": 'which kind',
    "input_attr": "x" # 'pos'
}

def classifier_init(model):
    #! load model
    #! or train model
    #! or do nothing
    return model

inherent_explainer = InherentModel(explainerconfig, model=model)
posthoc_explainer = PosthocExplainer(explainerconfig, model=model, classifier_init=classifier_init)

explainer = None
data_set = HEPDataset()
data = data_set[0]




# for post-hoc method, 传 classifier_init()
posthoc_explainer.train(data, pretrained_model_init=None)


# for some post_hoc method, directly use explain
posthoc_explainer.explain(data)
# for some post_hoc method, use train and then explain
posthoc_explainer.train(data)
posthoc_explainer.explain(data)

# for inherent method, use train and then explain
inherent_explainer.train(data_set)
inherent_explainer.explain(data_set)

prediction = inherent_explainer.predict(data)

#! in explain() we need model.get_emb(data), model.get_pred_from_emb(emb) and self.extractor(emb)
data_with_interpretation = inherent_explainer.explain(data)


class XEvaluator():
    def __init__(self, explainer) -> None:
        self.explainer = explainer

    def eval():
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

sensitive_eval = XEvaluator("sensitive", explainer=explainer)

fidel = sensitive_eval(data, data_with_interpretation)

decisive_eval = XEvaluator('decisive')

auc = decisive_eval(data_with_interpretation)



```
## Installation
```
pip install xgdl
```
