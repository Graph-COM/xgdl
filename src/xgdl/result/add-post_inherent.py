from pathlib import Path
import pandas as pd

def calculate_df(df):
    df[['exp_method', 'clf_method']] = df.index.str.split('_', n=1).tolist()

    # 修正clf_method列，因为它包含了额外的数字
    df[['clf_method', 'seed']] = df['clf_method'].str.rsplit('_', n=1).tolist()

    # 计算每一组的平均值和标准差
    result = df.groupby(['model', 'dataset', 'clf_method', 'exp_method']).agg(
        avg_exp_auc=('exp_auc', 'mean'),
        std_exp_auc=('exp_auc', 'std')
    ).reset_index()
    return result

# 定义一个函数来处理文件和目录
def process_directory(directory_path):
    all_data = []  # 用于存储所有csv文件数据的列表

    # 使用Path对象遍历目录
    path = Path(directory_path)
    
    # 遍历第一级目录，即model_dataset目录
    for model_dataset_dir in path.iterdir():
        if model_dataset_dir.is_dir():
            # 从目录名中提取model和dataset名称
            model, dataset = model_dataset_dir.name.split('_')
            
            # 遍历第二级目录，即explainer_inherent目录
            for explainer_inherent_dir in model_dataset_dir.iterdir():
                if explainer_inherent_dir.is_dir() and explainer_inherent_dir.name == 'explainer_inherent':
                    # 遍历CSV文件
                    for csv_file in explainer_inherent_dir.glob('*.csv'):
                        # 读取CSV文件到DataFrame
                        df = pd.read_csv(csv_file, index_col=0)
                        # 添加model和dataset列
                        df['model'] = model
                        df['dataset'] = dataset
                        # 将DataFrame添加到列表中
                        all_data.append(df)
    
    # 合并所有DataFrame
    final_df = pd.concat(all_data)
    return final_df

# 调用函数并传入顶级目录路径
df = process_directory('./result')
new_df = df[['test_exp_auc', 'model', 'dataset']].rename(columns={'test_exp_auc': 'exp_auc'})
final_df = calculate_df(new_df)

# reindex and multicolumn and combine results
final_df['avg_std'] = final_df.apply(lambda row: f"${row['avg_exp_auc']*100:.2f} \pm {row['std_exp_auc']:.2f}$", axis=1)
pivot_df = final_df.pivot_table(index=['clf_method', 'exp_method'], columns=['dataset', 'model'], values=["avg_std"], aggfunc='first')

# output final table
pd.set_option('display.max_columns', None)
print(pivot_df)
pivot_df.to_csv('./new-post-inherent.csv', index=True)


