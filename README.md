# XGDL
XGDL (eXplainability for Geometric Deep Learning) is an explainability library for scientific tasks using geometric deep learning.
______________________________________________________________________
## Quick Tour
```python
from xgdl import PosthocMethod, InherentModel, HEPDataset, XEvaluator

dataset = HEPDataset()
sample = dataset[0]

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


inherent_explainer = InherentModel(explainerconfig, model=model)

# for post-hoc method, we need func pretrained_model_init
def pretrained_model_init(model):
    #! load model
    #! or train model
    #! or do nothing
    return model
posthoc_explainer = PosthocExplainer(explainerconfig, model=model, pretrained_model_init=pretrained_model_init)




# for some post_hoc method, directly use explain
posthoc_explainer.explain(sample)

# for some post_hoc method, use train and then explain
posthoc_explainer.train(dataset)
posthoc_explainer.explain(sample)

# for inherent method, use train and then explain
inherent_explainer.train(dataset)
inherent_explainer.explain(sample)


#! in explain() we need model.get_emb(data), model.get_pred_from_emb(emb) and self.extractor(emb)
interpretation = inherent_explainer.explain(data)


sensitive_eval = XEvaluator("sensitive", explainer=explainer)
fidel = sensitive_eval(interpretation)


decisive_eval = XEvaluator('decisive')
auc = decisive_eval(interpretation)



```
## Installation
```
pip install xgdl
```
