import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据
data_file_path = '2026_MCM_Problem_C_Data.csv'
season11_file_path = 'season11_weekly_detail_table.csv'

# 加载数据
df_data = pd.read_csv(data_file_path)
df_season11 = pd.read_csv(season11_file_path)

# 提取职业和成绩（rank_sum_rank）
df_industry = df_data[['celebrity_name', 'celebrity_industry']]
df_scores = df_season11[['celebrity_name', 'rank_sum_rank']]

# 将职业进行标签编码
label_encoder = LabelEncoder()
df_industry['industry_encoded'] = label_encoder.fit_transform(df_industry['celebrity_industry'].fillna('Unknown'))  # 填充缺失值

# 合并数据：根据姓名进行合并
df_merged = pd.merge(df_industry, df_scores, on='celebrity_name', how='inner')

# 检查列名
print("Merged Data Columns:", df_merged.columns)

# 独热编码职业变量（直接对原始职业列进行独热编码）
onehot_encoder = OneHotEncoder(sparse_output=False)  # 使用新的参数名
industry_onehot = onehot_encoder.fit_transform(df_merged[['celebrity_industry']])

# 将独热编码结果转化为DataFrame
industry_onehot_df = pd.DataFrame(industry_onehot, columns=onehot_encoder.categories_[0])

# 检查列数是否匹配
print("Shape of the OneHotEncoded Data:", industry_onehot_df.shape)

# 结果变量（成绩）
y = df_merged['rank_sum_rank']

# 自变量（职业的独热编码）
X = industry_onehot_df

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 线性回归模型
regressor = LinearRegression()

# 训练模型
regressor.fit(X_train, y_train)

# 获取回归系数
coefficients = regressor.coef_

# 显示每个职业对成绩的回归系数
coefficients_df = pd.DataFrame(coefficients, index=industry_onehot_df.columns, columns=['Coefficient'])
print("\nLinear Regression Coefficients by Industry:")
print(coefficients_df)

# 测试模型预测效果
y_pred = regressor.predict(X_test)

# 输出模型的评分（决定系数R^2）
r2_score = regressor.score(X_test, y_test)
print("\nR^2 Score of the Model:", r2_score)
