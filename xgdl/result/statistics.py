import pandas as pd
import numpy as np
from pathlib import Path

# 初始化一个列表来存储每个文件的test_clf_auc值
def test():
    test_precision_values = []


    def collect_files_in_path(path_name):
        path = Path(path_name)
        


    # 遍历文件名
    for i in range(10):  # 假设文件名从0到9
        file_name = f"/nethome/jzhu617/renaissance/result/pointtrans_plbind/lri_gaussian/bs{i}_lri_gaussian_{i}.csv"  # 构造文件名
        df = pd.read_csv(file_name)  # 读取CSV文件
        # 假设每个文件只有一行数据，使用iloc[0]来获取这行数据
        # 如果有多行，可以根据具体情况调整
        test_precision = df['test_precision@20'].iloc[0]  
        test_precision_values.append(test_precision)  # 将值添加到列表中

    # 计算平均值和标准差
    avg_test_precision = np.mean(test_precision_values)
    std_precision_values = np.std(test_precision_values)

    print(f"Average test_clf_auc: {avg_test_precision}")
    print(f"Standard deviation of test_clf_auc: {std_precision_values}")


import pandas as pd
import os
import re

def process_csv_files(directory_path):
    # 初始化一个空的DataFrame来存储所有数据
    combined_df = pd.DataFrame()

    # 遍历指定目录下的所有文件
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".csv"):
            # 读取CSV文件
            file_path = os.path.join(directory_path, file_name)
            df = pd.read_csv(file_path, index_col=0)  # 假设第一列是索引
            
            # 从行索引中提取方法名并作为新的一列
            df['Method'] = [re.match(r"(.*?)_\d+", idx).group(1) for idx in df.index]
            
            # 将当前DataFrame添加到总的DataFrame中
            combined_df = pd.concat([combined_df, df])

    # 计算每个方法的test_clf_auc的平均值和标准差
    results = combined_df.groupby('Method')['test_precision@20'].agg(['mean', 'std']).reset_index()
    results.columns = ['Method', 'Avg_test_precision', 'Std_test_precision']  # 重命名列以更清晰地表示它们的含义

    return results

# 指定目录路径
import sys
backbone = sys.argv[1]
print(sys.argv[1])
directory_path = f'/usr/scratch/jzhu617/renaissance/result/{backbone}_plbind/all_inherent'
# 调用函数并打印结果
results_df = process_csv_files(directory_path)
print(results_df)

