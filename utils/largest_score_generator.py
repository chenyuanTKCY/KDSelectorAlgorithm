import pandas as pd

def main():
    # 加载CSV文件
    file_path = 'mergedTable_AUC_ROC.csv'  # 根据实际路径调整
    df = pd.read_csv(file_path)
    data = pd.read_csv(file_path, usecols=list(range(1, 13)))  # 读取第2到第14列（忽略第一列）

    # 确保所有AUC-ROC分数列都是数值类型，并处理数据中的异常或非数值
    for col in data.columns[1:12]:  # 调整为从第2列到第12列是AUC-ROC分数
        data[col] = pd.to_numeric(data[col], errors='coerce')  # 转换为数值，非数值转为NaN

    # 填充NaN值，确保nlargest可以正确执行
    data.fillna(value=0, inplace=True)  # 你可以根据需要更改填充策略

    # 定义一个函数来找到每一行中的前三大AUC-ROC分数
    def top_three_scores(row):
        scores = row[1:12]  # 正确索引分数列
        top_scores = scores.nlargest(3)
        return pd.Series(top_scores.values)

    # 应用函数并创建新列
    df[['1st_max', '2nd_max', '3rd_max']] = data.apply(top_three_scores, axis=1)

    # 将更新后的DataFrame保存到新的CSV文件
    df.to_csv('updated_mergedTable_AUC_ROC_2.csv', index=False)

    print("更新后的表格已保存。")

if __name__ == "__main__":
    main()