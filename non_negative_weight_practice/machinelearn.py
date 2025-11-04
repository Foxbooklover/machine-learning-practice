import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# PenDigits 데이터셋
digits = fetch_openml('pendigits', version=1, as_frame=False)
X = digits.data.astype(np.float32)
y = digits.target.astype(int)  # 숫자 레이블

# 4-bit binary로 변환
def int_to_bitvector(y, n_bits=4):
    return np.array([list(np.binary_repr(i, width=n_bits)) for i in y], dtype=np.float32)

y_bits = int_to_bitvector(y, n_bits=4)

# train/test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y_bits, test_size=0.2, random_state=42)

# torch tensor로 변환
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        last_size = input_size
        for h in hidden_sizes:
            self.layers.append(nn.Linear(last_size, h))
            last_size = h
        self.relu = nn.ReLU()
        self.output = nn.Linear(last_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.sigmoid(self.output(x))
        return x

input_size = 16
hidden_sizes = [9,7]
output_size = 4
model = MLP(input_size, hidden_sizes, output_size)


criterion = nn.BCELoss()  # 0/1 출력용
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 30
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# 모델 정의 후 학습 완료 가정
for name, param in model.named_parameters():
    print(name, param.shape)
    print(param.data)
