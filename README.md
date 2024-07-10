# XGDL 
<h1 align="center">Learnable Randomness Injection (LRI)</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2210.16966"><img src="https://img.shields.io/badge/-arXiv-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
    <a href="https://github.com/Graph-COM/LRI"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="https://openreview.net/forum?id=6u7mf9s2A9"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=ICLR%2723&color=blue"> </a>
    <!-- <a href="https://github.com/Graph-COM/GSAT/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/Graph-Com/GSAT?color=blue"> </a>
    <a href="https://icml.cc/virtual/2022/spotlight/17430"> <img src="https://img.shields.io/badge/Video-grey?logo=Kuaishou&logoColor=white" alt="Video"></a>
    <a href="https://icml.cc/media/icml-2022/Slides/17430.pdf"> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a>
    <a href="https://icml.cc/media/PosterPDFs/ICML%202022/a8acc28734d4fe90ea24353d901ae678.png"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a> -->
</p>

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

## Reference

If you find our paper and repo useful, please cite our relevant paper:
```bibtex
@misc{zhu2024understanding,
      title={Towards Understanding Sensitive and Decisive Patterns in Explainable AI: A Case Study of Model Interpretation in Geometric Deep Learning}, 
      author={Jiajun Zhu and Siqi Miao and Rex Ying and Pan Li},
      year={2024},
      eprint={2407.00849},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.00849}, 
}

@article{miao2023interpretable,
  title       = {Interpretable Geometric Deep Learning via Learnable Randomness Injection},
  author      = {Miao, Siqi and Luo, Yunan and Liu, Mia and Li, Pan},
  journal     = {International Conference on Learning Representations},
  year        = {2023}
}
```
