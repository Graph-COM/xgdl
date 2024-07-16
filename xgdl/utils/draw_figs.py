import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
from scipy.stats import pearsonr, linregress
plt.style.use(['science', 'notebook'])
import os
model_var_path = r"C:\Users\86173\Desktop\NEWMAIN\model_var\\"
save_path = r"C:\Users\86173\Desktop\NEWMAIN"


def generate_reliability_df(name_list, path, backbone_name):
    df = pd.DataFrame(columns=['Model', 'Backbone', 'quantification', 'ken_corr', 'spr_corr', 'pr_corr', 'avg_auc'])
    init_df = pd.read_csv(f"{path}/label_perturb1.csv", index_col=0)
    init_df['ACCfidpos'] = (init_df['ACCfidpos'] + 1) / 2
    for row_name in init_df.index:
        com_df = pd.DataFrame(columns=['exp_auc', 'ACCfidpos'])
        for name in name_list:
            csv_name = path + '\\' + name + '.csv'
            method_df = pd.read_csv(csv_name, index_col=0)
            method_df['ACCfidpos'] = (method_df['ACCfidpos'] + 1) / 2

            com_df.loc[len(com_df)] = [method_df.loc[row_name, 'exp_auc'], method_df.loc[row_name, 'ACCfidpos']]
        ken, spr, pr = com_df.corr('kendall').iloc[1, 0], com_df.corr('spearman').iloc[1, 0], com_df.corr('pearson').iloc[1, 0]
        avg_exp_auc = com_df['exp_auc'].mean()
        quantification = init_df.loc[row_name, 'ACCfidpos']
        df.loc[len(df)] = [row_name, backbone_name, quantification, ken, spr, pr, avg_exp_auc]
    return df

def generate_df_from_method_list(name_list, column, path, dataset_name):
    df = pd.DataFrame(columns=['Method', 'variable', 'exp_auc'])
    if dataset_name == 'synmol':
        int_list = [750, 800, 850, 900, 950, 1000]
        variables = [0.775, 0.825, 0.875, 0.925, 0.975]
    elif dataset_name == 'actstrack':
        int_list = [500, 600, 700, 800, 900, 1000]
        variables = [0.550, 0.650, 0.750, 0.850, 0.950]
    elif dataset_name == 'tau3mu':
        int_list = [650, 690, 730, 770, 810, 850]
        variables = [0.670, 0.710, 0.750, 0.790, 0.830]
    # for name in name_list:
    for name in name_list:
        csv_name = path + '\\' + name + '.csv'
        method_df = pd.read_csv(csv_name, index_col=0)
        method_df = method_df[~method_df.index.duplicated()]
        data_series = method_df[column]
        count = 0
        for index_name in method_df.index:
            auc = int(index_name[:3])
            for i in range(1, len(int_list)):
                if int_list[i-1] < auc <= int_list[i]:
                    df.loc[len(df)] = [name, variables[i - 1], data_series[index_name]]
                    # new_row = pd.DataFrame({'Method': name, 'variable': variables[i - 1], 'exp_auc': data_series[index_name]}, index=[0])
                    # df = pd.concat([df, new_row], ignore_index=True)
    return df


# plot notions https://www.notion.so/Notions-for-plot-0793de0cd418443786c80e0fcca30ad1
def draw_boxplot(ax, backbone, dataset, method_list, column, data_path, xlabel='Classification AUC', ylabel='Explanation AUC', legend=True):
    # plot
    # if 'exp' in column:
    # elif 'fid' in column:
    #     node_type = column[-3:]
    #     ylabel = f'Fidelity on {node_type.capitalize()} Samples'

    data = generate_df_from_method_list(method_list, column, data_path, dataset_name=dataset)
    sns.boxplot(data=data, x='variable', y='exp_auc', hue='Method', ax=ax,
                        showfliers=True,
                        showmeans=False,
                        # meanprops={'markerfacecolor': 'C0', 'markeredgecolor': 'white', 'marker': 'o'},
                        medianprops={"color": "white", "linewidth": 0.5},
                        boxprops={"edgecolor": "white", "linewidth": 0.5},
                        whiskerprops={"linewidth": 1.5},
                        capprops={"linewidth": 1.5})
    # VP = ax.boxplot(data,
    #                 patch_artist=True,
    #                 showfliers=True,
    #                 showmeans=True,
    #                 meanprops={'markerfacecolor': 'C0', 'markeredgecolor': 'white', 'marker': 'o'},
    #                 medianprops={"color": "white", "linewidth": 0.5},
    #                 boxprops={"facecolor": "C0", "edgecolor": "white",
    #                           "linewidth": 0.5},
    #                 whiskerprops={"color": "C0", "linewidth": 1.5},
    #                 capprops={"color": "C0", "linewidth": 1.5})
    ax.set(xlabel=xlabel, ylabel=ylabel, title=backbone.upper())
    if legend is False:
        ax.legend([])


def draw_box_for_one_dataset(axes, method_list, backbones, dataset):
    dataset_path = model_var_path + dataset + '/'
    ylabels = ["Explanation AUC", '', '']
    xlabels = ['', 'Classification AUC', '']
    legends = [True, False, False]
    for i, backbone in enumerate(backbones):
        draw_boxplot(axes[i], backbone, dataset, method_list, 'exp_auc', xlabel=xlabels[i], ylabel=ylabels[i], legend=legends[i], data_path=dataset_path+backbone)
    return


def corr_sensitivity_fidelity(value, path='C:\\Users\86173\Desktop\\0710collect\synmol\egnn\post-hoc'):
    df = pd.DataFrame()
    for root, ds, fs in os.walk(path):
        for f in fs:
            seed = f[5] # like bseed0
            fullname = os.path.join(root, f)
            data = pd.read_csv(fullname, index_col=0)
            sub_indexes = [i for i in data.index if i[-1] == seed or 'grad' in i or 'lrp' in i or 'label' in i ]
            data = data.loc[sub_indexes]
            data.set_index('exp_method', inplace=True)

            # partial dataframe
            df[seed] = data[value]
    print(df.T.corr('pearson'))
    print(df.T.corr('kendall'))
    print(df.T.corr('spearman'))


def draw_reliability_interval_box(axs, dataset, ordered_backbones, base_path=r'C:\Users\86173\Desktop\NEWMAIN\model_rely', coef='pearson', y='reliability'):
    dataset_path = f'{base_path}/{dataset}'
    df_list = []
    for i, backbone in enumerate(ordered_backbones):
        one_path = f'{dataset_path}/{backbone}'
        methods_list = name_list_1 if y == 'auc' else name_list_1 + ['label_perturb1']
        one_df = generate_reliability_df(methods_list, one_path, backbone)
        df = one_df.set_index('Model')
        # try:
        df["q_range"] = pd.cut(df['quantification'], bins=3, precision=2)
        # except:
        print()
        df_list += [df]
    all_df = pd.concat(df_list)
    y_name = 'avg_auc' if y == 'auc' else 'ken_corr' if coef == 'kendall' else 'spr_corr' if coef == 'spearman' else 'pr_corr'
    y_label = 'Average Explanation ROC AUC' if y == 'auc' else f'{coef.capitalize()} Correlation'
    all_df.sort_values(by=['Backbone', 'q_range'], inplace=True, ascending=[False, True])
    sns.boxplot(data=all_df, x='q_range', y=y_name, hue='Backbone', ax=axs,
                showfliers=True,
                showmeans=False,
                # meanprops={'markerfacecolor': 'C0', 'markeredgecolor': 'white', 'marker': 'o'},
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"edgecolor": "white", "linewidth": 0.5},
                whiskerprops={"linewidth": 1.5},
                capprops={"linewidth": 1.5})
    axs.tick_params(axis='x', labelrotation=30)
    axs.set(xlabel='Value Interval', ylabel=y_label, title=dataset.upper())


def draw_reliability_plots(axs, dataset, ordered_backbones, base_path=r'C:\Users\86173\Desktop\NEWMAIN\model_rely', coef='pearson', y='reliability'):
    dataset_path = f'{base_path}/{dataset}'
    df_list = []
    y_name = 'avg_auc' if y == 'auc' else 'ken_corr' if coef == 'kendall' else 'spr_corr' if coef == 'spearman' else 'pr_corr'
    y_label = 'Average Explanation ROC AUC' if y == 'auc' else f'{coef.capitalize()} Correlation'
    for i, backbone in enumerate(ordered_backbones):
        one_path = f'{dataset_path}/{backbone}'
        methods_list = name_list_1 if y == 'auc' else name_list_1+['label_perturb1']
        one_df = generate_reliability_df(methods_list, one_path, backbone)
        df = one_df.set_index('Model')

        axs.scatter(x=df['quantification'], y=df[y_name], label=backbone)
        # df["q_range"] = pd.cut(df['quantification'], bins=4, labels=[1, 2, 3, 4])
        df_list += [df]
    all_df = pd.concat(df_list)
    use_df = all_df[['quantification', y_name]]
    # text = f"Spearman coefficient: {use_df.corr('spearman').iloc[1, 0]:.2f}"
    # axs.text(0.05, 0.95, text, ha='left', va='top', transform=axs.transAxes)
    axs.legend()
        # sns.boxplot(data=all_df, x='q_range', y='pr_corr', hue='Backbone', ax=ax,
        #             showfliers=True,
        #             showmeans=False,
        #             # meanprops={'markerfacecolor': 'C0', 'markeredgecolor': 'white', 'marker': 'o'},
        #             medianprops={"color": "white", "linewidth": 0.5},
        #             boxprops={"edgecolor": "white", "linewidth": 0.5},
        #             whiskerprops={"linewidth": 1.5},
        #             capprops={"linewidth": 1.5})
    axs.set(xlabel='Label Fidelity AUC', ylabel=y_label, title=dataset.upper())


if __name__ == '__main__':
    backbones = ['egnn', 'dgcnn', 'pointtrans']
    name_list_1 = ['gradcam', 'inter_grad', 'gradx', 'gnnlrp']
    ordered_name_list = [name_list_1]
    datasets = ['synmol', 'actstrack', 'tau3mu']

    def explanation_prediction_corr():
        fig, axs = plt.subplots(nrows=len(datasets), ncols=3, sharex='row', sharey='row', figsize=(24, 6*len(datasets)))
        for index, dataset in enumerate(datasets):
            # corr_sensitivity_fidelity('ACCfid-pos')  # ['ACCfid+pos', 'ACCfid-pos', 'ACCfidpos']
            draw_box_for_one_dataset(axs[index], name_list_1, backbones, dataset)
        fig.tight_layout()
        fig.savefig(f'{save_path}/explanation_prediction_corr.pdf', dpi=300, transparent=True)
    # draw_boxplot(name, 'valid_fid_pos', post_fix='png')
    # explanation_prediction_corr()

    def reliability_value_quantification(datasets, type=('scatter', 'box'), y_axis='auc'):
        if 'scatter' in type:
            fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=(8*len(datasets), 6 * 1))
            for index, dataset in enumerate(datasets):
                draw_reliability_plots(axs[index], dataset, ordered_backbones=['pointtrans', 'egnn', 'dgcnn'], coef='pearson', y=y_axis)
            fig.tight_layout()
            fig.savefig(f'{save_path}/sca_reliability.pdf', dpi=300, transparent=True)
        if 'box' in type:
            fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=(8*len(datasets), 6 * 1))
            for index, dataset in enumerate(datasets):
                draw_reliability_interval_box(axs[index], dataset, ordered_backbones=['pointtrans', 'egnn', 'dgcnn'], coef='pearson', y=y_axis)
            fig.tight_layout()
            fig.savefig(f'{save_path}/box_reliability.pdf', dpi=300, transparent=True)

    reliability_value_quantification(['synmol', 'actstrack'])

