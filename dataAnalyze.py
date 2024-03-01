import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 指定文件路径
file_path = r'E:\PyCharm Community Edition 2023.2.1\MachineLearning\Dataset\train.csv'
# 读取CSV文件
df = pd.read_csv(file_path)

# 读取第3列到第5列的数据
x = df.iloc[:, 2:5].select_dtypes(include=[np.number]).values
# 读取第2列数据
y = df.iloc[:, 1]
# 检查y的数据类型，如果不是数值类型，则进行转换
if not np.issubdtype(y.dtype, np.number):
    y = y.astype(float)
y = y.values
# 将数据转换为张量
x = torch.tensor(x, dtype=torch.float32).to('cuda')
y = torch.tensor(y, dtype=torch.float32).to('cuda')
# 将数据转换为张量数据集
dataset = TensorDataset(x, y)
# 将数据集转换为数据加载器
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
# 定义模型并移动到GPU
model = nn.Linear(1, 1).to('cuda')
# 定义损失函数
criterion = nn.MSELoss()
# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 训练模型
    for epoch in range(100):
        for x, y in dataloader:
            # 前向传播
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 使用所有的数据进行预测
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 关闭梯度计算
        y_pred = model(x.to('cuda')).cpu()  # 将数据移动到GPU上进行预测，然后将预测结果移动到CPU上

# 绘制图像
plt.figure(i)
plt.plot(y.cpu(), 'r')  # 绘制真实值
plt.plot(y_pred, 'b')  # 绘制预测值
plt.legend(['True', 'Predicted'])
plt.show()

model.train()  # 设置模型为训练模式