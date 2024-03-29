import pandas as pd
import os

# 设定文件夹路径
folder_path = './data/text'

# CSV文件的输出路径和文件名
output_train_csv_path = './data/train_lighthouse.csv'
output_test_csv_path = './data/test_lighthouse.csv'

# 初始化一个空的DataFrame
df = pd.DataFrame(columns=['sentence', 'labels'])
tagList = []

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):  # 确保只处理.txt文件
        file_path = os.path.join(folder_path, file_name)

        # 读取txt文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            tags = lines[0].strip().split(',')  # 第一行是tags
            tags = [tag.strip() for tag in tags]  # 去除每个tag两边的空白字符

            if tags.__contains__(""):
                tags.remove("")

            for tag in tags:
                tagList.append(tag)

            # 读取剩下的每一行作为句子
            for sentence in lines[1:]:
                sentence = sentence.strip()
                if sentence:  # 确保句子不是空的
                    # 将数据添加到DataFrame中
                    df = df._append({'sentence': sentence, 'labels': tags}, ignore_index=True)

# 分割数据集为训练集和测试集 80:20比例
train_df = df.sample(frac=0.8, random_state=1)
test_df = df.drop(train_df.index)

# 将DataFrame保存到csv文件
train_df.to_csv(output_train_csv_path, index=False, sep='ㅇ')
test_df.to_csv(output_test_csv_path, index=False, sep='ㅇ')

# 获取并打印唯一标签列表
tagList = list(set(tagList))
print(tagList)
