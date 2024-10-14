# %%
### 데이터 처리
import pandas as pd
import numpy as np

### 데이터 스케일링 처리
from sklearn.preprocessing import StandardScaler

### 데이터 분류
from sklearn.model_selection import train_test_split

### 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc("font", family = "Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = True

### 상관관계
from scipy.stats import pearsonr

### 하이퍼파라메터 튜닝 자동화 클래스(모델)
from sklearn.model_selection import GridSearchCV

### 특성 생성 클래스(모델)
from sklearn.preprocessing import PolynomialFeatures

### 앙상블모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

### 가우시안 모델
from sklearn.mixture import GaussianMixture

### 평가 라이브러리
# 정확도
from sklearn.metrics import accuracy_score
# 정밀도(긍정적 오차 적용)
from sklearn.metrics import precision_score
# 재현율(부적정 오차 적용)
from sklearn.metrics import recall_score
# F1-score (긍정/부정 모두 적용)
from sklearn.metrics import f1_score

### 오차행렬(혼동행렬) 매트릭스 및 시각화
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
### 군집모델
from sklearn.cluster import KMeans

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import sklearn.metrics as metric
from sklearn import preprocessing
import torch.nn as nn


# %%
file_path = "./Press_RawDataSet.xlsx"
press_data = pd.read_excel(file_path)
press_data

# %%
press_data.drop(['idx', 'Machine_Name', 'Item No'], axis=1, inplace=True)
press_data.drop(['working time'], axis=1, inplace=True)
press_data.dropna(axis=0, inplace=True)

# %%
press_data

# %%
press_data.info()

# %%
for feature in press_data :
    print(feature, press_data[feature].value_counts())

# %%
press_data.describe()

# %%
press_data.iloc[:,:-1].corr()

# %%
bin=[15, 15, 25, 13, 17, 15, 35]
for index, value in enumerate(press_data.iloc[:,:-1].columns) :
    plt.figure(index)
    plt.hist(press_data.iloc[:,:-1][value], bins=bin[index], facecolor=(144/255,171/255,221/255), linewidth=3, edgecolor='black')
    plt.title(value)

# %%
n_press_data = press_data.iloc[:,:-1]
# n_press_data.iloc[:,1:]
n_press_data

# %%
scaler = preprocessing.MinMaxScaler()
scaler.fit(n_press_data.iloc[:,1:])
scaled_data = scaler.fit_transform(n_press_data.iloc[:,1:])

# %%
class AutoEncoder(nn.Module) :
    def __init__(self, input_size, hidden_size, output_size):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.hiiden_size = hidden_size
        self.output_size = output_size

        self.AutoEncoder = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.RReLU(),
            nn.Linear(hidden_size[0], output_size),
            nn.RReLU(),
            nn.Linear(output_size, hidden_size[0]),
            nn.RReLU(),
            nn.Linear(hidden_size[0], output_size)
        )
    def forward(self, inputs) :
        output = self.AutoEncoder(inputs)
        
        return output
        

# %%
# class AutoEncoder(nn.Module) :
#     def __init__(self, input_size, hidden_size, output_size):
#         super(AutoEncoder, self).__init__()
#         self.input_size = input_size
#         self.hiiden_size = hidden_size
#         self.output_size = output_size

#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, hidden_size[0]),
#             nn.RReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_size[0], input_size),  # 입력 크기와 동일하게
#             nn.RReLU()
#         )
#     def forward(self, inputs) :
#         encoded = self.encoder(inputs)
#         decoded = self.decoder(encoded)
#         return decoded

# %%
train_data = torch.Tensor(scaled_data[:45051])
print(len(train_data))
test_data = torch.Tensor(scaled_data[45051:])
print(len(test_data))

# %%
epoch = 5
batch_size = 64
lr = 0.01

input_size = len(train_data[0])
hidden_size = [3]
output_size = input_size

criterion = nn.MSELoss()
optimizer = torch.optim.Adam

AutoEncoder = AutoEncoder(input_size, hidden_size, output_size)

# %%
def train_net(AutoEncoder, data, criterion, epochs, lr_rate=0.01) :
    optim = optimizer(AutoEncoder.parameters(), lr=lr_rate)
    data_iter = DataLoader(data, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs +1) :
        running_loss = 0.0
        for x in data_iter :
            optim.zero_grad()
            output = AutoEncoder(x)
            loss = criterion(x, output)
            loss.backward()
            optim.step()
            running_loss += loss.item()

        print("epoch{}, loss: {:.2f}".format(epoch, running_loss))
    return AutoEncoder

# %%
AutoEncoder = train_net(AutoEncoder, train_data, criterion, epoch, lr)

# %%
for x in DataLoader(train_data, batch_size=batch_size):
    output = AutoEncoder(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    break


# %%
train_loss_chart = []
for data in train_data :
    output = AutoEncoder(data)
    loss = criterion(output, data)
    train_loss_chart.append(loss.item())

threshold = np.mean(train_loss_chart) + np.std(train_loss_chart)*8
print("Threshold :", threshold)

# %%
test_loss_chart = []
for data in train_data :
    output = AutoEncoder(data)
    loss = criterion(output, data)
    test_loss_chart.append(loss.item())

outlier = list(test_loss_chart >= threshold)
outlier.count(True)

# %%



