import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 设置随机种子以保持结果一致性
torch.manual_seed(0)
np.random.seed(0)

# 创建一个简单的非线性数据集
n = 200  # 样本数量
X = (np.random.rand(n, 1) * 80) - 20  # 生成-20到60之间的随机数
y = np.sin(X) + np.random.randn(n, 1) * 0.2  # 非线性关系加上噪声

# 将numpy数组转换为PyTorch张量，并移动到GPU
X_tensor = torch.from_numpy(X).float().cuda()
y_tensor = torch.from_numpy(y).float().cuda()

# 划分数据集
cv = np.random.rand(n) < 0.7
X_train = X_tensor[cv]
y_train = y_tensor[cv]
X_test = X_tensor[~cv]
y_test = y_tensor[~cv]

# 定义模型
class SinModel(nn.Module):
    def __init__(self):
        super(SinModel, self).__init__()
        self.b1 = nn.Parameter(torch.randn(1, requires_grad=True).cuda())
        self.b2 = nn.Parameter(torch.randn(1, requires_grad=True).cuda())
        self.b3 = nn.Parameter(torch.randn(1, requires_grad=True).cuda())
        self.b4 = nn.Parameter(torch.randn(1, requires_grad=True).cuda())

    def forward(self, x):
        return self.b1 * torch.sin(self.b2 * x + self.b3) + self.b4

# 初始化模型并移动到GPU
model = SinModel().cuda()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 在测试集上评估模型
with torch.no_grad():
    y_test_pred = model(X_test)
    test_error = criterion(y_test_pred, y_test)
print(f'测试集上的均方误差: {test_error.item()}')

# 绘制数据集和拟合曲线
plt.figure()
plt.plot(X, y, 'bo', label='原始数据')

# 提取 X_tensor 的最小和最大值作为标量
X_min = X_tensor.min().item()
X_max = X_tensor.max().item()

# 使用标量创建线性空间的张量，并移动到 GPU
X_fit = torch.linspace(X_min, X_max, 100).unsqueeze(1).cuda()

# 在模型上运行 X_fit 并将结果转换为 NumPy 数组
y_fit = model(X_fit).detach().cpu().numpy()
plt.plot(X_fit.cpu().numpy(), y_fit, 'r-', label='拟合曲线')
plt.xlabel('X')
plt.ylabel('y')
plt.title('正弦回归拟合')
plt.legend()
plt.show()
