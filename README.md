# XGDL 
XGDL (eXplainability for Geometric Deep Learning) is an explainability library for scientific tasks using geometric deep learning. (The interface is in a state of ongoing enhancement.)
______________________________________________________________________
## Features
- The implementation of 13 methods including self-interpretable (inherent) and post-hoc methods
- The evaluation pipeline for both sensitive and deicisve patterns (see our paper for more details)
- The dataloader module for scientific datasets.

## Quick Tour

### Load Dataset
```python
from xgdl import ScienceDataset    

dataset = ScienceDataset.from_name('synmol')
key_subset = ScienceDataset.filter_signal_class(dataset)
sample = key_subset[0]
```

> Output: Data(x=[18, 1], y=[1, 1], pos=[18, 3], node_label=[18], mol_df_idx=[1], edge_index=[2, 90])

### Use Self-interpretatble Model
```python
from xgdl import InherentModel

inherent_config = {
        'method': "lri_bern",
        'model': "egnn", # choose from ['egnn', 'dgcnn', 'pointtrans']
        "dataset": "synmol", # choose from ['synmol', 'tau3mu', 'actstrack', 'plbind']
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
            },
        "training":
            {
            'clf_lr': 1.0e-3,
            'clf_wd': 1.0e-5,
            'exp_lr': 1.0e-3,
            'exp_wd': 1.0e-5,
            'batch_size': 4,
            'epoch': 1,
            }
        }

inherent_explainer = InherentModel(inherent_config)

# for inherent method, use train and then explain
inherent_explainer.train(dataset)
interpretation = inherent_explainer.explain(sample)

```
### Use Post-hoc Method
```python
from xgdl import PosthocMethod

posthoc_config = {
    'method': "gradcam",
    'model': "egnn", # choose from ['egnn', 'dgcnn', 'pointtrans']
    "dataset": "synmol", # choose from ['synmol', 'tau3mu', 'actstrack', 'plbind']
    # "train_from_scratch": True,
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
    },
    "training":
        {
        'clf_lr': 1.0e-3,
        'clf_wd': 1.0e-5,
        'exp_lr': 1.0e-3,
        'exp_wd': 1.0e-5,
        'batch_size': 4,
        'epoch': 1,
        'warmup': 1,
        }
}

posthoc_explainer = PosthocMethod(posthoc_config)

# for post_hoc method of class PostAttributor, omit train and directly explain
posthoc_explainer.train(dataset)
interpretation = posthoc_explainer.explain(sample)
```

### Evaluate Model Interpretation

```python
print(interpretation)
```
> Output: Data(x=[20, 1], y=[1, 1], pos=[20, 3], node_label=[20], mol_df_idx=[1], edge_index=[2, 100], node_imp=[20])

```python
from xgdl import XEvaluator 

sensitive_eval = XEvaluator("sensitive", pretrained_model_init=pretrained_model_init)
fidel = sensitive_eval(interpretation)

decisive_eval = XEvaluator('decisive')
auc = decisive_eval(interpretation)

```

## Installation
```
pip install xgdl
```
