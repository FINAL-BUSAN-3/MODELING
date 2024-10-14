# %% [markdown]
# ### 라이브러리 및 함수 정의

# %%
### 선정된 5개 모델
# Poly ElasticNet
# MSE: 0.001253
# Test Score: 0.846528
# Train Score: 0.864578
# Fit Score: 0.018050

# GradientBoosting(lr=0.01)
# MSE: 0.001161
# Test Score: 0.857814
# Train Score: 0.866020
# Fit Score: 0.008206

# Voting Ensemble
# MSE: 0.001681
# Test Score: 0.794179
# Train Score: 0.798907
# Fit Score: 0.004728

# LinearRegression
# MSE: 0.002337
# Test Score: 0.713832
# Train Score: 0.725186
# Fit Score: 0.011355

# Ridge(alpha=0.1)
# MSE: 0.002337
# Test Score: 0.713832
# Train Score: 0.725186
# Fit Score: 0.011355

# %%
# 튜토리얼 진행을 위한 모듈 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
import warnings
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"]="Malgun Gothic"
 
# 경고 메시지 출력 표기 생략
warnings.filterwarnings('ignore')

np.set_printoptions(suppress=True, precision=3)

# # SEED 설정 (본인이 조정 가능)
# SEED = 30



# %%
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor

# %%

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
# 1) 폴리노멀 퓨처를 (POM) 다중특성만들기를 해서..
# 2) Standard Scaling 후
# 3) degree = 2 or 3 으로 한 다음에 (특성을 늘릴 때의 차원...)
# 4) train, test 데이터 만들기

# %%


# %%
### 인코딩 함수
def label_encoder_fit(x):
    unique_values = np.unique(x)
    labels = {value: i for i, value in enumerate(unique_values)}
    return labels

def label_encoder_transform(x, labels):
    return [labels[value] for value in x]

### 스케일링 함수
def feature_scaling(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled


# %% [markdown]
# ### 데이터 전처리

# %%
file_path = "./Press_RawDataSet.xlsx"
press_data = pd.read_excel(file_path)
press_data.head(5)
press_data['working time'].value_counts()
press_data.drop(['idx', 'Machine_Name', 'Item No'], axis=1, inplace=True)
press_data.drop(['working time'], axis=1, inplace=True)
press_data.dropna(axis=0, inplace=True)

# %%
### 군집모델
from sklearn.cluster import KMeans
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
# train, test 데이터 나누기
from sklearn.model_selection import train_test_split

data=df.iloc[:,:-1].to_numpy()
target=df["gmm_cluster"].to_numpy()
data.shape, target.shape


# test_size = 0.4로 고정
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.4, random_state=42
)

# %%
# 특성 차원 늘리기
# degree = 2 해보고, degree=3 도 해서 비교해보면 좋음 
# (할 거면 안 헷갈리게 파일 구분이나 코드 구분 잘 해주세요)

poly = PolynomialFeatures(degree=2, include_bias=False)

poly.fit(x_train)

x_train = poly.transform(x_train)
x_test  = poly.transform(x_test)

print("특성 생성 패턴 : ", poly.get_feature_names_out())
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# %%
# 스케일링 : 스탠다드 사용
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# %%
    """
    여기까지 따로 수정할 부분 없습니다
    모델링부터 해주세요
    """

# %% [markdown]
# ### 모델링(릿지)

# %%
""" 
바꿔야 하는 부분

1) 모델 이름 
2) param_grid 
3) param_grid 개수에 맞춰서, best_params 각각 출력
    --> 예시 : best_ridge_alpha = best_ridge_params['alpha'] 
    => 이거 param 개수에 맞게 출력해야함
4)
"""

### 예시
# 모델 정의 (본인이 맡은 모델)
ridge = Ridge()

# 하이퍼파라미터 그리드 정의
# - 모델에 따라 바꾸기 (gpt한테 물어보면 잘 알려줌) 
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              # 절편을 사용할지 여부
              'fit_intercept': [True, False],
               # solver 알고리즘 선택
              'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']  }

# GridSearchCV를 사용하여 하이퍼파라미터 튜닝
# - ridge_grid 이름 바꿔주기
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# 모델 훈련
# - ridge_grid 이름 바꿔주기
grid_search.fit(x_train, y_train)

# 최적 하이퍼파라미터 및 최적 점수 출력
# 최적의 하이퍼파라미터 및 모델
best_params = grid_search.best_params_
print(f"Best parameters for GradientBoosting: {best_params}")

# 최적 모델로 예측 및 평가
# - ridge_grid 이름 바꿔주기
best_model = grid_search.best_estimator_
pred = best_model.predict(x_test)
train_score = best_model.score(x_train, y_train)
test_score = best_model.score(x_test, y_test)


# MSE / 점수 출력
# - 모델 제목은 센스있게 바꿔주세요 !  => @@@(best params)
mse_eval(f'Ridge(best params)', pred, y_test)
getscore(f'Ridge(best params)', train_score, test_score)


# %% [markdown]
# # GradientBoostingRegressor

# %%
# 모델 정의 (Gradientboost)
gradientboost = GradientBoostingRegressor()

# 하이퍼파라미터 그리드 정의
# - 모델에 따라 바꾸기 (gpt한테 물어보면 잘 알려줌) 
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],  # 학습률 조정
    'n_estimators': [100, 500, 1000],  # 트리의 개수
    'max_depth': [3, 5, 7],  # 트리의 최대 깊이
    'min_samples_split': [2, 5, 10],  # 내부 노드를 분할하기 위한 최소 샘플 수
    'min_samples_leaf': [1, 2, 4],  # 리프 노드에 있어야 하는 최소 샘플 수
    'subsample': [0.7, 0.8, 1.0],  # 데이터 샘플링 비율
    'max_features': ['auto', 'sqrt', 'log2']  # 사용할 최대 피처 수
}


# GridSearchCV를 사용하여 하이퍼파라미터 튜닝
# - ridge_grid 이름 바꿔주기
grid_search = GridSearchCV(estimator=gradientboost, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# 모델 훈련
# - ridge_grid 이름 바꿔주기
grid_search.fit(x_train, y_train)

# 최적 하이퍼파라미터 및 최적 점수 출력
# 최적의 하이퍼파라미터 및 모델
best_params = grid_search.best_params_
print(f"Best parameters for GradientBoosting: {best_params}")

# 최적 모델로 예측 및 평가
# - ridge_grid 이름 바꿔주기
best_model = grid_search.best_estimator_
pred = best_model.predict(x_test)
train_score = best_model.score(x_train, y_train)
test_score = best_model.score(x_test, y_test)


# MSE / 점수 출력
# - 모델 제목은 센스있게 바꿔주세요 !  => @@@(best params)
mse_eval(f'GradientBoost(best params)', pred, y_test)
getscore(f'GradientBoost(best params)', train_score, test_score)


# %% [markdown]
# # Voting Ensemble 회귀

# %%
# 개별 모델 정의
ridge = Ridge()
gbr = GradientBoostingRegressor()
tree = DecisionTreeRegressor()

# Voting Regressor 구성
voting_regressor = VotingRegressor(estimators=[
    ('ridge', ridge),
    ('gbr', gbr),
    ('tree', tree)
])

# 개별 모델 하이퍼파라미터 그리드 정의
param_grid = {
    # Ridge의 하이퍼파라미터 튜닝
    'ridge__alpha': [0.01, 0.1, 1, 10],
    
    # GradientBoostingRegressor의 하이퍼파라미터 튜닝
    'gbr__n_estimators': [100, 500, 1000],
    'gbr__learning_rate': [0.001, 0.01, 0.1],
    'gbr__max_depth': [3, 5, 7],
    
    # DecisionTreeRegressor의 하이퍼파라미터 튜닝
    'tree__max_depth': [3, 5, 7],
    
    # Voting Ensemble의 가중치 튜닝 (가중치가 더 큰 모델은 더 큰 영향력)
    'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]  # Ridge, GBR, Tree 순으로 가중치 조정
}

# GridSearchCV를 사용한 하이퍼파라미터 튜닝
grid_search = GridSearchCV(estimator=voting_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# 모델 훈련
# - ridge_grid 이름 바꿔주기
grid_search.fit(x_train, y_train)

# 최적 하이퍼파라미터 및 최적 점수 출력
# 최적의 하이퍼파라미터 및 모델
best_params = grid_search.best_params_
print(f"Best parameters for VotingRegressor: {best_params}")

# 최적 모델로 예측 및 평가
# - ridge_grid 이름 바꿔주기
best_model = grid_search.best_estimator_
pred = best_model.predict(x_test)
train_score = best_model.score(x_train, y_train)
test_score = best_model.score(x_test, y_test)


# MSE / 점수 출력
# - 모델 제목은 센스있게 바꿔주세요 !  => @@@(best params)
mse_eval(f'VotingRegressor(best params)', pred, y_test)
getscore(f'VotingRegressor(best params)', train_score, test_score)


# %% [markdown]
# # LinearRegression

# %%
# 파이프라인: StandardScaler로 데이터 정규화 후 LinearRegression 적용
pipe = make_pipeline(StandardScaler(), LinearRegression())

# 설정할 하이퍼파라미터 그리드
param_grid = {
    'linearregression__fit_intercept': [True, False],  # 절편을 사용할지 여부
    'linearregression__copy_X': [True, False]  # 원본 데이터의 복사 여부 (속도 관련)
}

# GridSearchCV를 사용한 하이퍼파라미터 튜닝
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# 모델 훈련
grid_search.fit(x_train, y_train)

# 최적 하이퍼파라미터 및 최적 점수 출력
best_params = grid_search.best_params_
print(f"Best parameters for LinearRegression: {best_params}")

# 최적 모델로 예측 및 평가
best_model = grid_search.best_estimator_
pred = best_model.predict(x_test)
train_score = best_model.score(x_train, y_train)
test_score = best_model.score(x_test, y_test)

# MSE / 점수 출력
mse_eval(f'LinearRegression({best_params})', pred, y_test)
getscore(f'LinearRegression({best_params})', train_score, test_score)


# %% [markdown]
# # Poly ElasticNet

# %%
# 설정할 하이퍼파라미터 그리드
param_grid = {
    'polynomialfeatures__degree': [2, 3, 4],  # 다항식 차수
    'elasticnet__alpha': [0.001, 0.01, 0.1, 1, 10],  # ElasticNet 규제 강도
    'elasticnet__l1_ratio': [0.2, 0.5, 0.7, 0.9, 1.0]  # L1 규제와 L2 규제 간의 비율
}

# Polynomial ElasticNet 모델 생성
poly_elasticnet = make_pipeline(PolynomialFeatures(), ElasticNet())

# GridSearchCV를 사용한 하이퍼파라미터 튜닝
grid_search = GridSearchCV(estimator=poly_elasticnet, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# 모델 훈련
# - ridge_grid 이름 바꿔주기
grid_search.fit(x_train, y_train)

# 최적 하이퍼파라미터 및 최적 점수 출력
# 최적의 하이퍼파라미터 및 모델
best_params = grid_search.best_params_
print(f"Best parameters for Poly ElasticNet: {best_params}")

# 최적 모델로 예측 및 평가
# - ridge_grid 이름 바꿔주기
best_model = grid_search.best_estimator_
pred = best_model.predict(x_test)
train_score = best_model.score(x_train, y_train)
test_score = best_model.score(x_test, y_test)


# MSE / 점수 출력
# - 모델 제목은 센스있게 바꿔주세요 !  => @@@(best params)
mse_eval(f'Poly ElasticNet(best params)', pred, y_test)
getscore(f'Poly ElasticNet(best params)', train_score, test_score)

# %% [markdown]
# # Voting Ensemble 분류

# %%
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 개별 모델 정의
log_reg = LogisticRegression()
tree_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()

# VotingClassifier 구성
voting_clf = VotingClassifier(estimators=[
    ('lr', log_reg),
    ('dt', tree_clf),
    ('rf', rf_clf)
], voting='soft')

# 하이퍼파라미터 그리드 설정
param_grid = {
    # Logistic Regression의 하이퍼파라미터
    'lr__C': [0.01, 0.1, 1, 10],  # 정규화 강도

    # Decision Tree의 하이퍼파라미터
    'dt__max_depth': [3, 5, 7],  # 결정 트리의 최대 깊이

    # Random Forest의 하이퍼파라미터
    'rf__n_estimators': [50, 100, 200],  # 랜덤 포레스트에서 트리의 개수
    'rf__max_depth': [5, 10, 20],  # 랜덤 포레스트에서 각 트리의 최대 깊이

    # Voting 가중치 (각 모델의 중요도 설정)
    'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]],  # Logistic, Decision Tree, Random Forest 순으로 가중치 설정
}

# GridSearchCV를 사용한 하이퍼파라미터 튜닝
grid_search = GridSearchCV(estimator=voting_clf, param_grid=param_grid, cv=5, scoring='accuracy')

# 모델 훈련
# - ridge_grid 이름 바꿔주기
grid_search.fit(x_train, y_train)

# 최적 하이퍼파라미터 및 최적 점수 출력
# 최적의 하이퍼파라미터 및 모델
best_params = grid_search.best_params_
print(f"Best parameters for VotingClassifier: {best_params}")

# 최적 모델로 예측 및 평가
# - ridge_grid 이름 바꿔주기
best_model = grid_search.best_estimator_
pred = best_model.predict(x_test)
train_score = best_model.score(x_train, y_train)
test_score = best_model.score(x_test, y_test)


# MSE / 점수 출력
# - 모델 제목은 센스있게 바꿔주세요 !  => @@@(best params)
mse_eval(f'Poly VotingClassifier(best params)', pred, y_test)
getscore(f'Poly VotingClassifier(best params)', train_score, test_score)

# %%
df

# %%



