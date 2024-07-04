import argparse
import numpy as np
from nni.experiment import Experiment, ExperimentConfig, AlgorithmConfig, LocalConfig, RemoteConfig, RemoteMachineConfig


def main():
    parser = argparse.ArgumentParser(description='Tuning')
    parser.add_argument('--name', type=str, required=True, help='name of the experiment')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', required=True)
    parser.add_argument('-m', '--method', type=str, help='method used', required=True)
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', required=True)
    args = parser.parse_args()


    if args.method == 'lri_bern':
        search_space = {
            'epochs': {'_type': 'choice', '_value': [50]},
            'info_loss_coef': {'_type': 'choice', '_value': [0.1, 1.0, 0.01, 10]},
            'final_r': {'_type': 'choice', '_value': [0.5, 0.7, 0.3]},
        }

    elif args.method == 'vgib':
        search_space = {
            'mi_loss_coef': {'_type': 'choice', '_value': [1e-3, 1e-4, 1e-5]},
            'reg_loss_coef': {'_type': 'choice', '_value': [0.1, 1, 10]},
            'noise_loss_coef': {'_type': 'choice', '_value': [10, 1, 0.1]},
        }

    elif args.method == 'ciga':
        search_space = {
            # 'warmup': {'_type': 'choice', '_value': [0, 100, 200]},
            'contrast_loss_coef': {'_type': 'choice', '_value': [0.1, 1, 10]},
            'hinge_loss_coef': {'_type': 'choice', '_value': [0.01, 0.1, 1]},
            'causal_ratio': {'_type': 'choice', '_value': [0.6, 0.7, 0.8]},
        }


    elif args.method == 'lri_gaussian':
        pos_coef_options = {
            'actstrack': [200.0, 300.0],  # [100.0, 200.0, 300.0]  # 1 is ~20%, 10%, 5% percentiles
            'synbind': [10.0, 15.0],  # [5.0, 10.0, 15.0]  # 1 is ~10%, 7%, 5% percentiles
            'tau3mu': [7, 11],  # [3, 7, 11]  # 1 is ~10%, 5%, 3% percentiles
            'plbind': [0.9, 1.2]  # [0.6, 0.9, 1.2]  # 1 is ~10%, 6.5%, 5% percentiles
        }
        kr_options = {
            'actstrack': [10],  # [5, 10]
            'synbind': [5],  # [5, 10]
            'tau3mu': [1.5],  # [1.0, 1.5]
            'plbind': [7]  # [5, 8]
        }

        search_space = {
            'dataset': {'_type': 'choice', '_value': ['acts_2T']},
            'epochs': {'_type': 'choice', '_value': [100]},


            'pred_lr': {'_type': 'choice', '_value': [1.0e-3]},
            'info_loss_coef': {'_type': 'choice', '_value': [0.01, 0.1, 1.0]},

            'pos_coef': {'_type': 'choice', '_value': pos_coef_options[args.dataset]},
            'kr': {'_type': 'choice', '_value': kr_options[args.dataset]},
        }

    elif args.method == 'subgraphx':
        search_space = {
            'subgraph_building_method': {'_type': 'choice', '_value': ['split', 'zero_filling']},
            'high2low': {'_type': 'choice', '_value': [True, False]},
            'score_threshold': {'_type': 'choice', '_value': [0.1, 0.2, 0.3]}
        }

    elif args.method == 'asap':
        search_space = {
            'casual_ratio': {'_type': 'choice', '_value': [0.4, 0.5, 0.6]},
        }

    elif args.method == 'pgmexplainer':
        search_space = {
            'pred_threshold': {'_type': 'choice', '_value': [0, 0.3, 0.5]},
            'percentage': {'_type': 'choice', '_value': [20, 40, 60]},
            'perturb_mode': {'_type': 'choice', '_value': ['split']}
        }

    elif args.method == 'pgexplainer':
        search_space = {
            'epoch': {'_type': 'choice', '_value': [200]},
            # 'dataset': {'_type': 'choice', '_value': ['acts_2T', 'synbind']},
            'size_loss_coef': {'_type': 'choice', '_value': [0.01]},
            'mask_ent_loss_coef': {'_type': 'loguniform', '_value': [0.001, 1]},
        }

    elif args.method == 'gnnexplainer':
        search_space = {
            'size_loss_coef': {'_type': 'choice', '_value': [1.0]},
            'mask_ent_loss_coef': {'_type': 'choice', '_value': [0.01, 0.1]},
            'iter_lr': {'_type': 'uniform', '_value': [0.4, 0.6]}
        }
    else:
        raise ValueError('Unknown method: {}'.format(args.method))


    # if args.dataset == 'all_acts':
    #     assert args.backbone == 'dgcnn'
    #     if args.method == 'psat':
    #         search_space = {
    #             'seed': {'_type': 'choice', '_value': list(range(5))},
    #             'dataset': {'_type': 'choice', '_value': [f'acts_{each}T' for each in range(2, 22, 2)]},
    #             'data/feature_type': {'_type': 'choice', '_value': ['only_pos']},
    #             f'model/{args.backbone}/dropout_p': {'_type': 'choice', '_value': [0.0]},
    #
    #             'psat/warmup': {'_type': 'choice', '_value': [100]},
    #             'psat/epochs': {'_type': 'choice', '_value': [500]},
    #             'psat/pred_lr': {'_type': 'choice', '_value': [1.0e-8]},
    #             'psat/one_gnn': {'_type': 'choice', '_value': [False]},
    #             'psat/info_loss_coef': {'_type': 'choice', '_value': [10.0]},
    #             'psat/pos_coef': {'_type': 'choice', '_value': [100.0]},
    #             'psat/param_u': {'_type': 'choice', '_value': ['sig_dense']},
    #             'psat/dropout_p': {'_type': 'choice', '_value': [0.0]},
    #             'psat/covar_dim': {'_type': 'choice', '_value': [2.0]}
    #         }
    #     else:
    #         assert args.method == 'gradcam'
    #         search_space = {
    #             'seed': {'_type': 'choice', '_value': list(range(5))},
    #             'dataset': {'_type': 'choice', '_value': [f'acts_{each}T' for each in range(2, 22, 2)]},
    #             'data/feature_type': {'_type': 'choice', '_value': ['only_pos']},
    #             f'model/{args.backbone}/dropout_p': {'_type': 'choice', '_value': [0.0]},
    #
    #             'gradcam/warmup': {'_type': 'choice', '_value': [100]},
    #             'gradcam/epochs': {'_type': 'choice', '_value': [1]},
    #             # 'gradcam/target_layers': {'_type': 'choice', '_value': ['last']},
    #             # 'gradcam/way_to_sum': {'_type': 'choice', '_value': ['plain']},
    #             'gradcam/grad_pos': {'_type': 'choice', '_value': [True]},
    #             'gradcam/way_to_sum_pos': {'_type': 'choice', '_value': ['norm']},
    #         }

    trial_per_gpu = 2
    gpu_ratio = 1 / trial_per_gpu
    gpu_index = [3, 4, 5]
    command = f'python trainer.py -d {args.dataset} -m {args.method} -b {args.backbone} --cuda 99'
    config = ExperimentConfig(
        debug=True,
        experiment_name=args.name,
        trial_command=command,
        trial_code_directory='.',
        trial_gpu_number=1,
        trial_concurrency=trial_per_gpu*(len(gpu_index)+1),
        search_space=search_space,
        max_trial_number=int(np.prod([len(v['_value']) for k, v in search_space.items()])),
        experiment_working_directory='../nni-experiments',
        assessor=AlgorithmConfig(name='Medianstop'),
        tuner=AlgorithmConfig(name='GridSearch', class_args={"optimize_mode": 'maximize'}),
        # experiment.config.tuner.name = 'SMAC'
        # experiment.config.max_trial_number = 10000
        training_service=LocalConfig(use_active_gpu=True, max_trial_number_per_gpu=trial_per_gpu, gpu_indices=gpu_index)
        # training_service=RemoteConfig(machine_list=[RemoteMachineConfig(use_active_gpu=True, max_trial_number_per_gpu=\
        # trial_per_gpu, gpu_indices=[0, 1, 2, 3, 4], host='130.207.232.51', user='jzhu617',password='024convdreamStf') ])
        )

    experiment = Experiment(config)
    experiment.run(8082)
    input('Press enter to quit')
    experiment.stop()
    # experiment.view()


if __name__ == '__main__':
    main()