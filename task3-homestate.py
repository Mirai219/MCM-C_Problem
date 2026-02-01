import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 加载数据
data_file_path = '2026_MCM_Problem_C_Data.csv'
season11_file_path = 'season11_weekly_detail_table.csv'

# 加载数据
df_data = pd.read_csv(data_file_path)
df_season11 = pd.read_csv(season11_file_path)

# 提取家乡和成绩（rank_sum_rank）
df_home = df_data[['celebrity_name', 'celebrity_homestate']]
df_scores = df_season11[['celebrity_name', 'rank_sum_rank']]

# 合并数据：根据姓名进行合并
df_merged = pd.merge(df_home, df_scores, on='celebrity_name', how='inner')

# 使用 pd.get_dummies() 对家乡进行独热编码
df_home_dummies = pd.get_dummies(df_merged['celebrity_homestate'], prefix='home')

# 结果变量（成绩）
y = df_merged['rank_sum_rank']

# 自变量（家乡的独热编码）
X = df_home_dummies

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 线性回归模型
regressor = LinearRegression()

# 训练模型
regressor.fit(X_train, y_train)

# 获取回归系数
coefficients = regressor.coef_

# 显示每个家乡对成绩的回归系数
coefficients_df = pd.DataFrame(coefficients, index=X.columns, columns=['Coefficient'])
print("\nLinear Regression Coefficients by Home State:")
print(coefficients_df)

# 测试模型预测效果
y_pred = regressor.predict(X_test)

# 输出模型的评分（决定系数R^2）
r2_score = regressor.score(X_test, y_test)
print("\nR^2 Score of the Model:", r2_score)
