from xgdl import ScienceDataset    

dataset = ScienceDataset.from_name('synmol')
sample = dataset[0]
print(sample)

from xgdl import InherentModel

config = {
    'method': "lri_bern",
    'model': "egnn", # choose from []
    "dataset": "tau3mu",
    "hyperparameter":
        {
        'pred_loss_coef': 0.1,
        'info_loss_coef': 0.05,
        'temperature': 1.0,
        'final_r': 0.9,
        'decay_interval': 10,
        'decay_r': 0.01,
        'init_r': 0.5,
        'attn_constraint': True
        }
    }

inherent_explainer = InherentModel.from_config(config, dataset_info=dataset)
print(inherent_explainer)
# for inherent method, use train and then explain
inherent_explainer.train(dataset)

inherent_explainer.explain(sample)

#! in explain() we need model.get_emb(data), model.get_pred_from_emb(emb) and self.extractor(emb)
interpretation = inherent_explainer.explain(sample)

from xgdl import PosthocMethod

config = {
    'name': "gradcam",
    'clf': 1,
    'extractor': 1,
    'criterion': 1,

    'pred_loss_coef': 0.1,
    'info_loss_coef': 0.05,
    'temperature': 1.0,
    'final_r': 0.9,
    'decay_interval': 10,
    'decay_r': 0.01,
    'init_r': 0.5,
    'attn_constraint': True
}

# for post-hoc method, we need func pretrained_model_init
def pretrained_model_init(model):
    #! load model
    #! or train model
    #! or do nothing
    return model

posthoc_explainer = PosthocMethod(config, pretrained_model_init=pretrained_model_init)

# for some post_hoc method, directly use explain
posthoc_explainer.explain(sample)

# for some post_hoc method, use train and then explain
posthoc_explainer.train(dataset)
posthoc_explainer.explain(sample)

### Evaluate Model Interpretation


from xgdl import XEvaluator 

intepretation = ...

sensitive_eval = XEvaluator("sensitive", pretrained_model_init=pretrained_model_init)
fidel = sensitive_eval(interpretation)

decisive_eval = XEvaluator('decisive')
auc = decisive_eval(interpretation)