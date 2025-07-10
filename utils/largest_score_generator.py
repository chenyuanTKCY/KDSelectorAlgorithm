import pandas as pd

def main():
    file_path = 'mergedTable_AUC_ROC.csv'
    df = pd.read_csv(file_path)
    data = pd.read_csv(file_path, usecols=list(range(1, 13)))


    for col in data.columns[1:12]:
        data[col] = pd.to_numeric(data[col], errors='coerce')


    data.fillna(value=0, inplace=True)


    def top_three_scores(row):
        scores = row[1:12]
        top_scores = scores.nlargest(3)
        return pd.Series(top_scores.values)

    df[['1st_max', '2nd_max', '3rd_max']] = data.apply(top_three_scores, axis=1)

    df.to_csv('updated_mergedTable_AUC_ROC_2.csv', index=False)

    print("更新后的表格已保存。")

if __name__ == "__main__":
    main()