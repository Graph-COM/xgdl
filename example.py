from xgdl import ScienceDataset    
from xgdl import InherentModel

dataset = ScienceDataset.from_name('synmol')
key_subset = ScienceDataset.filter_signal_class(dataset)
sample = key_subset[0]

def test_inherent():
    config = {
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
    
    inherent_explainer = InherentModel(config)
    print(inherent_explainer)

    # for inherent method, use train and then explain
    inherent_explainer.train(dataset)

    #! in explain() we need model.get_emb(data), model.get_pred_from_emb(emb) and self.extractor(emb)
    interpretation = inherent_explainer.explain(sample)
    return interpretation

def main():
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

    # for some post_hoc method, directly use explain
    # posthoc_explainer.explain(sample)

    # for some post_hoc method, use train and then explain
    posthoc_explainer.train(dataset)
    interpretation = posthoc_explainer.explain(sample)
    print(interpretation)

    ## Evaluate Model Interpretation

    from xgdl import XEvaluator 

    intepretation = ...

    sensitive_eval = XEvaluator("sensitive")
    fidel = sensitive_eval(interpretation)

    decisive_eval = XEvaluator('decisive')
    auc = decisive_eval(interpretation)

if __name__ == '__main__':
    main()