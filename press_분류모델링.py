# %% [markdown]
# ### 라이브러리

# %%
# 튜토리얼 진행을 위한 모듈 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
import warnings

# 경고 메시지 출력 표기 생략
warnings.filterwarnings('ignore')

np.set_printoptions(suppress=True, precision=3)

# SEED 설정
SEED = 30

# %%
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

my_predictions = {}
my_pred = None
my_actual = None
my_name = None

colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive', 
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray', 
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
         ]

def plot_predictions(name_, pred, actual):
    df = pd.DataFrame({'name': name_,'prediction': pred, 'actual': actual})
    df = df.sort_values(by='name').reset_index(drop=True)
    plt.figure(figsize=(11, 8))
    plt.scatter(df.index, df['prediction'], marker='x', color='r')
    plt.scatter(df.index, df['actual'], alpha=0.7, marker='o', color='black')
    plt.title(name_, fontsize=15)
    plt.legend(['prediction', 'actual'], fontsize=12)
    plt.show()

def mse_eval(name_, pred, actual):
    global my_predictions, colors, my_pred, my_actual, my_name
    
    my_name = name_
    my_pred = pred
    my_actual = actual

    plot_predictions(name_, pred, actual)

    mse = mean_squared_error(pred, actual)
    my_predictions[name_] = mse

    y_value = sorted(my_predictions.items(), key=lambda x: x[1], reverse=True)
    
    df = pd.DataFrame(y_value, columns=['model', 'mse'])
    df = df.sort_values(by='model').reset_index(drop=True)
    print(df)
    min_ = df['mse'].min() - 10
    max_ = df['mse'].max() + 10
    
    length = len(df) / 2
    plt.figure(figsize=(9, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=12)
    bars = ax.barh(np.arange(len(df)), df['mse'], height=0.3)
    
    for i, v in enumerate(df['mse']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=12, fontweight='bold', verticalalignment='center')
        
    plt.title('MSE Error', fontsize=16)
    plt.xlim(min_, max_)
    
    plt.show()
    
def add_model(name_, pred, actual):
    global my_predictions, my_pred, my_actual, my_name
    my_name = name_
    my_pred = pred
    my_actual = actual
    
    mse = mean_squared_error(pred, actual)
    my_predictions[name_] = mse

def remove_model(name_):
    global my_predictions
    try:
        del my_predictions[name_]
    except KeyError:
        return False
    return True

def plot_all():
    global my_predictions, my_pred, my_actual, my_name
    
    plot_predictions(my_name, my_pred, my_actual)
    
    y_value = sorted(my_name.items(), key=lambda x: x[1], reverse=True)
    
    df = pd.DataFrame(y_value, columns=['model', 'mse'])
    df = df.sort_values(by='model').reset_index(drop=True)
    print(df)
    min_ = df['mse'].min() - 10
    max_ = df['mse'].max() + 10
    
    length = len(df) / 2
    
    plt.figure(figsize=(9, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=12)
    bars = ax.barh(np.arange(len(df)), df['mse'], height=0.3)
    
    for i, v in enumerate(df['mse']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=12, fontweight='bold', verticalalignment='center')
        
    plt.title('MSE Error', fontsize=16)
    plt.xlim(min_, max_)
    
    plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

my_scores1 = {}
my_scores2 = {}
my_scores3 = {}
my_train_score  = None
my_test_score = None
my_name = None

colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive', 
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray', 
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
         ]
### pred= train_score
### actual = test_score
def makingscore(name_, train_score, test_score):
    df_score = pd.DataFrame.from_records([{'name': name_, 'train_score': train_score, 'test_score': test_score, 'fit_score': train_score-test_score}])
    df_score1 = df_score.sort_values(by='name').reset_index(drop=True)
    df_score2 = df_score.sort_values(by='name').reset_index(drop=True)
    df_score3 = df_score.sort_values(by='name').reset_index(drop=True)


def getscore(name_, train_score, test_score):
    global my_scores1, my_scores2, my_scores3, colors, my_train_score, my_test_score, my_name
    my_name = name_
    my_train_score = train_score
    my_test_score = test_score
    fit_score= train_score - test_score
    
    makingscore(my_name, my_train_score, my_test_score)
    # df_score = pd.DataFrame.from_records([{'train_score': train_score, 'test_score': test_score, 'fit_score': train_score-test_score}])
    # df_score1 = df_score.sort_values(by='train_score').reset_index(drop=True)
    # df_score2 = df_score.sort_values(by='test_score').reset_index(drop=True)
    # df_score3 = df_score.sort_values(by='fit_score').reset_index(drop=True)
    
    my_scores1[name_] = train_score
    y_value1 = sorted(my_scores1.items(), key=lambda x: x[1], reverse=True)
    
    df_score1= pd.DataFrame(y_value1, columns=['model', 'train_score'])
    df_score1 = df_score1.sort_values(by='model').reset_index(drop=True)

    print(df_score1)
    min_ = -1
    max_ = 1
    
    length = len(df_score1) / 2
    
    plt.figure(figsize=(9, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df_score1)))
    ax.set_yticklabels(df_score1['model'], fontsize=12)
    bars = ax.barh(np.arange(len(df_score1)), df_score1['train_score'], height=0.3)
    
    for i, v in enumerate(df_score1['train_score']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v+0.02 , i, str(round(v, 3)), color='k', fontsize=12, fontweight='bold', verticalalignment='center')
        
    plt.title('train_score', fontsize=16)
    plt.xlim(min_, max_)
    
    plt.show()
    
    my_scores2[name_] = test_score
    y_value2 = sorted(my_scores2.items(), key=lambda x: x[1], reverse=True)
    
    df_score2= pd.DataFrame(y_value2, columns=['model', 'test_score'])
    df_score2 = df_score2.sort_values(by='model').reset_index(drop=True)

    print(df_score2)
    min_ = -1
    max_ = 1
    
    length = len(df_score2) / 2
    
    plt.figure(figsize=(9, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df_score2)))
    ax.set_yticklabels(df_score2['model'], fontsize=12)
    bars = ax.barh(np.arange(len(df_score2)), df_score2['test_score'], height=0.3)
    
    for i, v in enumerate(df_score2['test_score']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v+0.02, i, str(round(v, 3)), color='k', fontsize=12, fontweight='bold', verticalalignment='center')
        
    plt.title('test_score', fontsize=16)
    plt.xlim(min_, max_)
    
    plt.show()
    
    my_scores3[name_] = fit_score
    y_value3 = sorted(my_scores3.items(), key=lambda x: x[1], reverse=True)
    
    df_score3 = pd.DataFrame(y_value3, columns=['model', 'fit_score'])
    df_score3 = df_score3.sort_values(by='model').reset_index(drop=True)

    print(df_score3)
    min_ = -1
    max_ = 1
    
    length = len(df_score3) / 2
    
    plt.figure(figsize=(9, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df_score3)))
    ax.set_yticklabels(df_score3['model'], fontsize=12)
    bars = ax.barh(np.arange(len(df_score3)), df_score3['fit_score'], height=0.3)
    
    for i, v in enumerate(df_score3['fit_score']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v+0.02, i, str(round(v, 3)), color='k', fontsize=12, fontweight='bold', verticalalignment='center')
        
    plt.title('fit_score', fontsize=16)
    plt.xlim(min_, max_)
    
    plt.show()
    
def add_model(name_, train_score, test_score):
    global my_scores1, my_scores2, my_scores3, colors, my_train_score, my_test_score, my_name, fit_score
    my_name = name_
    my_train_score = train_score
    my_test_score = test_score
    
    fit_score= train_score - test_score
    my_scores1[name_] = train_score
    my_scores2[name_] = test_score
    my_scores3[name_] = fit_score

def remove_model(name_):
    global my_scores1, my_scores2, my_scores3
    try:
        del my_scores1[name_]
        del my_scores2[name_]
        del my_scores3[name_]
        
    except KeyError:
        return False
    return True

# %% [markdown]
# ### 데이터 불러들이기 / 전처리

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

# %%
file_path = "./Press_RawDataSet.xlsx"
press_data = pd.read_excel(file_path)
press_data.head(5)
press_data['working time'].value_counts()
press_data.drop(['idx', 'Machine_Name', 'Item No'], axis=1, inplace=True)
press_data.drop(['working time'], axis=1, inplace=True)
press_data.dropna(axis=0, inplace=True)

# %%
scaler = StandardScaler()
input_data = scaler.fit_transform(press_data.iloc[:,:-1])

gmm = KMeans(n_clusters= 2, random_state=42,n_init=10)
gmm.fit(input_data)
gmm_labels = gmm.predict(input_data)

press_data['gmm_cluster'] = gmm_labels
press_data['gmm_cluster'].value_counts()

df = press_data
df

# %%
    """
    데이터셋의 첫 두 열에는 샘플의 고유 ID 번호와 진단 결과(M = 악성, B = 양성)가 들어 있습니다.
    3번째에서 32번째까지 열에는 세포 핵의 디지털 이미지에서 계산된 30개의 실수 값 특성이 담겨 있습니다. 
    이 특성을 사용하여 종양이 악성인지 양성인지 예측하는 모델을 만들 것입니다. 
    """

# %% [markdown]
# ### 데이터셋 처리

# %% [markdown]
# ##### 인코딩 함수

# %%
### 인코딩 함수
def label_encoder_fit(x):
    unique_values = np.unique(x)
    labels = {value: i for i, value in enumerate(unique_values)}
    return labels

def label_encoder_transform(x, labels):
    return [labels[value] for value in x]

# %% [markdown]
# ##### 스케일링 함수

# %%
### 스케일링 함수
def feature_scaling(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled

# %%


# %% [markdown]
# ### 인코딩

# %%
label_encoders = {}

# %%
### 인코딩 작업
for column in df.columns:
    if df[column].dtype == 'object':
        labels = label_encoder_fit(df[column])
        label_encoders[column] = labels
        df[column] = label_encoder_transform(df[column], labels)
label_encoders

# %%
df

# %% [markdown]
# ### 훈련/검증 데이터 만들기

# %%
from sklearn.model_selection import train_test_split

data=df.iloc[:,:-1].to_numpy()
target=df["gmm_cluster"].to_numpy()
data.shape, target.shape

x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.4, random_state=42
)


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# %% [markdown]
# ### 스케일링

# %%
x_train = feature_scaling(x_train)
x_test = feature_scaling(x_test)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Accuracy, Precision, Recall, F1 Score 시각화 함수
def prf_eval(name_, pred, actual, average='weighted'):
    global my_predictions
    
    # 초기화
    if 'my_predictions' not in globals():
        my_predictions = {}
    
    # Accuracy, Precision, Recall, F1 Score 계산
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average=average)
    recall = recall_score(actual, pred, average=average)
    f1 = f1_score(actual, pred, average=average)
    
    # Dictionary에 저장 (각 모델별로 성능 지표 저장)
    my_predictions[name_] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Debugging: print my_predictions to ensure the structure is correct
    print("my_predictions:", my_predictions)

    # 데이터 프레임 생성
    df = pd.DataFrame(my_predictions).T.reset_index()
    df.columns = ['model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Debugging: print the dataframe to check structure
    print("Generated DataFrame:")
    print(df)
    
    # 시각화
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    num_metrics = len(metrics_list)
    
    # Bar plot 그리기
    length = len(df) / 2
    plt.figure(figsize=(10, length * num_metrics))
    
    for i, metric in enumerate(metrics_list):
        plt.subplot(1, num_metrics, i + 1)
        ax = plt.gca()
        ax.set_title(f'{metric} by Model', fontsize=14)
        
        sorted_df = df.sort_values(by=metric).reset_index(drop=True)
        bars = ax.barh(sorted_df['model'], sorted_df[metric], color=np.random.rand(len(sorted_df), 3))

        # 각 바에 값 표시
        for j, v in enumerate(sorted_df[metric]):
            ax.text(v + 0.01, j, str(round(v, 3)), color='k', fontsize=10, verticalalignment='center')

        ax.set_xlim([0, 1])  # 모든 지표는 0~1 사이의 값
        ax.set_xlabel(metric, fontsize=12)
    
    plt.tight_layout()
    plt.show()


# %%
# 1. 로지스틱 회귀 (Logistic Regression)
# 2. K-최근접 이웃 (K-Nearest Neighbors, KNN)
# 3. 서포트 벡터 머신 (Support Vector Machine, SVM)
# 4. 결정 트리 (Decision Tree)
# 5. 랜덤 포레스트 (Random Forest)
# 6. 그래디언트 부스팅 (Gradient Boosting)
# 7. XGBoost
# 8. LightGBM
# 9. CatBoost
# 10. 나이브 베이즈 (Naive Bayes)
# 11. 다층 퍼셉트론 (Multi-Layer Perceptron, MLP)
# 12. 확률적 경사 하강법 (Stochastic Gradient Descent, SGDClassifier)
# 13. AdaBoost
# 14. Stacking Classifier
# 15. Bagging Classifier
# 16. Voting Classifier
# 17. Gaussian Process Classifier
# 18. Quadratic Discriminant Analysis (QDA)
# 19. Linear Discriminant Analysis (LDA)
# 20. Passive Aggressive Classifier

# %% [markdown]
# ### LogisticRegression

# %%
# Logistic Regression 코드
from sklearn.linear_model import LogisticRegression

# Logistic Regression 모델 생성
logistic_reg = LogisticRegression()

# 모델 학습
logistic_reg.fit(x_train, y_train)

# 예측 수행
pred = logistic_reg.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('LogisticRegression', pred, y_test)
# 학습 및 테스트 정확도
train_score = logistic_reg.score(x_train, y_train)
test_score = logistic_reg.score(x_test, y_test)

getscore('LogisticRegression', train_score, test_score)
# 학습 및 테스트 정확도 출력
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')

# %%
# K-Nearest Neighbors (KNN) 코드
from sklearn.neighbors import KNeighborsClassifier

# KNN 모델 생성 (기본 k=5)
knn = KNeighborsClassifier(n_neighbors=5)

# 모델 학습
knn.fit(x_train, y_train)

# 예측 수행
pred = knn.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('K-Nearest Neighbors', pred, y_test)

# 학습 및 테스트 정확도
train_score = knn.score(x_train, y_train)
test_score = knn.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('K-Nearest Neighbors', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Support Vector Machine (SVM) 코드
from sklearn.svm import SVC

# SVM 모델 생성 (기본 커널: 'rbf')
svm = SVC(kernel='rbf')

# 모델 학습
svm.fit(x_train, y_train)

# 예측 수행
pred = svm.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Support Vector Machine', pred, y_test)

# 학습 및 테스트 정확도
train_score = svm.score(x_train, y_train)
test_score = svm.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Support Vector Machine', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Decision Tree 코드
from sklearn.tree import DecisionTreeClassifier

# Decision Tree 모델 생성
decision_tree = DecisionTreeClassifier()

# 모델 학습
decision_tree.fit(x_train, y_train)

# 예측 수행
pred = decision_tree.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Decision Tree', pred, y_test)

# 학습 및 테스트 정확도
train_score = decision_tree.score(x_train, y_train)
test_score = decision_tree.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Decision Tree', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Random Forest 코드
from sklearn.ensemble import RandomForestClassifier

# Random Forest 모델 생성
random_forest = RandomForestClassifier()

# 모델 학습
random_forest.fit(x_train, y_train)

# 예측 수행
pred = random_forest.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Random Forest', pred, y_test)

# 학습 및 테스트 정확도
train_score = random_forest.score(x_train, y_train)
test_score = random_forest.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Random Forest', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Gradient Boosting 코드
from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting 모델 생성
gradient_boosting = GradientBoostingClassifier()

# 모델 학습
gradient_boosting.fit(x_train, y_train)

# 예측 수행
pred = gradient_boosting.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Gradient Boosting', pred, y_test)

# 학습 및 테스트 정확도
train_score = gradient_boosting.score(x_train, y_train)
test_score = gradient_boosting.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Gradient Boosting', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# XGBoost 코드
from xgboost import XGBClassifier

# XGBoost 모델 생성
xgboost_model = XGBClassifier()

# 모델 학습
xgboost_model.fit(x_train, y_train)

# 예측 수행
pred = xgboost_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('XGBoost', pred, y_test)

# 학습 및 테스트 정확도
train_score = xgboost_model.score(x_train, y_train)
test_score = xgboost_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('XGBoost', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# LightGBM 코드
from lightgbm import LGBMClassifier

# LightGBM 모델 생성
lightgbm_model = LGBMClassifier()

# 모델 학습
lightgbm_model.fit(x_train, y_train)

# 예측 수행
pred = lightgbm_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('LightGBM', pred, y_test)

# 학습 및 테스트 정확도
train_score = lightgbm_model.score(x_train, y_train)
test_score = lightgbm_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('LightGBM', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# CatBoost 코드
from catboost import CatBoostClassifier

# CatBoost 모델 생성
catboost_model = CatBoostClassifier(verbose=0)  # verbose=0으로 학습 과정에서 출력되지 않도록 설정

# 모델 학습
catboost_model.fit(x_train, y_train)

# 예측 수행
pred = catboost_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('CatBoost', pred, y_test)

# 학습 및 테스트 정확도
train_score = catboost_model.score(x_train, y_train)
test_score = catboost_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('CatBoost', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Naive Bayes 코드
from sklearn.naive_bayes import GaussianNB

# Naive Bayes 모델 생성
naive_bayes_model = GaussianNB()

# 모델 학습
naive_bayes_model.fit(x_train, y_train)

# 예측 수행
pred = naive_bayes_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Naive Bayes', pred, y_test)

# 학습 및 테스트 정확도
train_score = naive_bayes_model.score(x_train, y_train)
test_score = naive_bayes_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Naive Bayes', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Multi-Layer Perceptron (MLP) 코드
from sklearn.neural_network import MLPClassifier

# MLP 모델 생성
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# 모델 학습
mlp_model.fit(x_train, y_train)

# 예측 수행
pred = mlp_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Multi-Layer Perceptron', pred, y_test)

# 학습 및 테스트 정확도
train_score = mlp_model.score(x_train, y_train)
test_score = mlp_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Multi-Layer Perceptron', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Stochastic Gradient Descent (SGD) 코드
from sklearn.linear_model import SGDClassifier

# SGDClassifier 모델 생성
sgd_model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# 모델 학습
sgd_model.fit(x_train, y_train)

# 예측 수행
pred = sgd_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('SGDClassifier', pred, y_test)

# 학습 및 테스트 정확도
train_score = sgd_model.score(x_train, y_train)
test_score = sgd_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('SGDClassifier', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# AdaBoost 코드
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost 모델 생성
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)

# 모델 학습
adaboost_model.fit(x_train, y_train)

# 예측 수행
pred = adaboost_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('AdaBoost', pred, y_test)

# 학습 및 테스트 정확도
train_score = adaboost_model.score(x_train, y_train)
test_score = adaboost_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('AdaBoost', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Stacking Classifier 코드
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 기본 학습기 정의
estimators = [
    ('decision_tree', DecisionTreeClassifier()),
    ('svc', SVC(probability=True)),  # SVC의 확률 출력 사용
    ('knn', KNeighborsClassifier())
]

# StackingClassifier 모델 생성 (메타 모델로 LogisticRegression 사용)
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# 모델 학습
stacking_model.fit(x_train, y_train)

# 예측 수행
pred = stacking_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Stacking Classifier', pred, y_test)

# 학습 및 테스트 정확도
train_score = stacking_model.score(x_train, y_train)
test_score = stacking_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Stacking Classifier', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Bagging Classifier 코드
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# BaggingClassifier 모델 생성 (기본 학습기로 DecisionTreeClassifier 사용)
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)

# 모델 학습
bagging_model.fit(x_train, y_train)

# 예측 수행
pred = bagging_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Bagging Classifier', pred, y_test)

# 학습 및 테스트 정확도
train_score = bagging_model.score(x_train, y_train)
test_score = bagging_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Bagging Classifier', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Voting Classifier 코드
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 여러 기본 학습기 정의
estimators = [
    ('logistic', LogisticRegression()),
    ('decision_tree', DecisionTreeClassifier()),
    ('svc', SVC(probability=True)),
    ('knn', KNeighborsClassifier())
]

# VotingClassifier 모델 생성 (Soft Voting 사용)
voting_model = VotingClassifier(estimators=estimators, voting='soft')

# 모델 학습
voting_model.fit(x_train, y_train)

# 예측 수행
pred = voting_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Voting Classifier', pred, y_test)

# 학습 및 테스트 정확도
train_score = voting_model.score(x_train, y_train)
test_score = voting_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Voting Classifier', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Gaussian Process Classifier 코드
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Gaussian Process Classifier 모델 생성 (기본 커널로 RBF 사용)
gaussian_process_model = GaussianProcessClassifier(kernel=RBF())

# 모델 학습
gaussian_process_model.fit(x_train, y_train)

# 예측 수행
pred = gaussian_process_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Gaussian Process Classifier', pred, y_test)

# 학습 및 테스트 정확도
train_score = gaussian_process_model.score(x_train, y_train)
test_score = gaussian_process_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Gaussian Process Classifier', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# QDA 코드
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# QDA 모델 생성
qda_model = QuadraticDiscriminantAnalysis()

# 모델 학습
qda_model.fit(x_train, y_train)

# 예측 수행
pred = qda_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Quadratic Discriminant Analysis', pred, y_test)

# 학습 및 테스트 정확도
train_score = qda_model.score(x_train, y_train)
test_score = qda_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Quadratic Discriminant Analysis', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# LDA 코드
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA 모델 생성
lda_model = LinearDiscriminantAnalysis()

# 모델 학습
lda_model.fit(x_train, y_train)

# 예측 수행
pred = lda_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Linear Discriminant Analysis', pred, y_test)

# 학습 및 테스트 정확도
train_score = lda_model.score(x_train, y_train)
test_score = lda_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Linear Discriminant Analysis', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %%
# Passive Aggressive Classifier 코드
from sklearn.linear_model import PassiveAggressiveClassifier

# Passive Aggressive Classifier 모델 생성
passive_aggressive_model = PassiveAggressiveClassifier(max_iter=1000, random_state=42, tol=1e-3)

# 모델 학습
passive_aggressive_model.fit(x_train, y_train)

# 예측 수행
pred = passive_aggressive_model.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('Passive Aggressive Classifier', pred, y_test)

# 학습 및 테스트 정확도
train_score = passive_aggressive_model.score(x_train, y_train)
test_score = passive_aggressive_model.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('Passive Aggressive Classifier', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


# %% [markdown]
# # 하이퍼파라메터 튜닝

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# 로지스틱 회귀 모델 정의
model = LogisticRegression()

# 하이퍼파라미터 그리드 설정
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300]
}

# GridSearchCV를 사용한 튜닝
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

pred = grid_search.predict(x_test)

# 최적의 하이퍼파라미터 출력
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
# 평가 지표 계산 및 시각화
prf_eval('LogisticRegression 튜닝', pred, y_test)

# 학습 및 테스트 정확도
train_score = grid_search.score(x_train, y_train)
test_score = grid_search.score(x_test, y_test)

# 학습 및 테스트 정확도 출력
getscore('LogisticRegression 튜닝', train_score, test_score)
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')

# %%
# Logistic Regression 코드
from sklearn.linear_model import LogisticRegression

# Logistic Regression 모델 생성
logistic_reg = LogisticRegression()

# 모델 학습
logistic_reg.fit(x_train, y_train)

# 예측 수행
pred = logistic_reg.predict(x_test)

# 평가 지표 계산 및 시각화
prf_eval('LogisticRegression', pred, y_test)
# 학습 및 테스트 정확도
train_score = logistic_reg.score(x_train, y_train)
test_score = logistic_reg.score(x_test, y_test)

getscore('LogisticRegression', train_score, test_score)
# 학습 및 테스트 정확도 출력
print(f'Train Accuracy: {train_score}')
print(f'Test Accuracy: {test_score}')
print(f'Fit Accuracy: {train_score - test_score}')


