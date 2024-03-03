import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 读取训练数据
file_path = 'E:\\PyCharm Community Edition 2023.2.1\\MachineLearning\\Dataset\\train.csv'
df = pd.read_csv(file_path)
data = df.iloc[:, 7].values
target = df.iloc[:, 1].values

# 转换为PyTorch张量
train_data_tensor = torch.tensor(data, dtype=torch.float32)
train_target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

# 创建数据加载器
train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义模型
model = nn.Linear(3, 1)  # 输入特征数为3，输出特征数为1

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 训练模型
for epoch in range(100):
    for batch_data, batch_target in train_loader:
        # 前向传播
        output = model(batch_data)
        loss = criterion(output, batch_target)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空过往梯度
        loss.backward()  # 反向传播，计算当前梯度
        optimizer.step()  # 根据梯度更新网络参数

# 读取测试数据
test_file_path = 'E:\\PyCharm Community Edition 2023.2.1\\MachineLearning\\Dataset\\test.csv'
test_df = pd.read_csv(test_file_path)
test_data = test_df.iloc[:, 2:5].select_dtypes(include=[np.number]).values
test_target = test_df.iloc[:, 1].values

# 转换为PyTorch张量
test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
test_target_tensor = torch.tensor(test_target, dtype=torch.float32).unsqueeze(1)

# 测试模型
model.eval()  # 将模型设置为评估模式
with torch.no_grad():  # 在这个with下，所有计算得出的tensor都不会计算梯度，从而节约内存
    predictions = model(test_data_tensor)  # 测试数据前向传播
    test_loss = criterion(predictions, test_target_tensor)  # 计算测试损失

print(f'Test loss: {test_loss.item()}')  # 打印测试损失

# 绘制图像
plt.figure()
plt.plot(test_target_tensor.cpu().numpy(), 'r', label='True')  # 绘制真实值
plt.plot(predictions.detach().cpu().numpy(), 'b', label='Predicted')  # 绘制预测值
plt.legend()
plt.show()
