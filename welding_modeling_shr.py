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

# %%
file_path = "dataset/Welding Data Set_01.xlsx"
welding_data = pd.read_excel(file_path)
welding_data.head(5)
welding_data['working time'].value_counts()
welding_data.drop(['idx', 'Machine_Name', 'Item No'], axis=1, inplace=True)
welding_data.drop(['working time'], axis=1, inplace=True)
welding_data.dropna(axis=0, inplace=True)

# %%
scaler = StandardScaler()
input_data = scaler.fit_transform(welding_data.iloc[:,:-1])

gmm = GaussianMixture(n_components= 4, random_state=42)
gmm.fit(input_data)
gmm_labels = gmm.predict(input_data)

welding_data['gmm_cluster'] = gmm_labels
welding_data['gmm_cluster'].value_counts()

df = welding_data
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

# %% [markdown]
# ### 로지스틱 회귀분석 함수 (최적화)

# %%
# Logistic regression functions
def predict(X,W,b):
    return W.dot(X)+b

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(W,b):
    m,n = train_scaled_input.shape
    cost = 0
    for i in range(m):
        fx = sigmoid(predict(train_scaled_input[i], W, b))
        cost += train_target[i]*np.log(fx)+(1-train_target[i])*np.log(1-fx)
    cost = -cost/m
    return cost

def gradient_step(W, b):
    m,n = train_scaled_input.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        fx = sigmoid(predict(train_scaled_input[i], W, b))
        for j in range(n):
            dj_dw[j] += (fx - train_target[i]) * train_scaled_input[i][j]
        dj_db += fx - train_target[i]
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return [dj_dw, dj_db]

def gradient_decent(alpha, iterations, init_W, init_b, record_interval):
    local_W = init_W
    local_b = init_b
    cost_history = [cost(local_W, local_b)]
    for i in range(iterations):
        new_W, new_b = gradient_step(local_W, local_b)
        local_W = local_W - (alpha*new_W)
        local_b = local_b - (alpha*new_b)
        if i%record_interval==0:
            local_cost = cost(local_W, local_b)
            print(f"Iteration {i}: Cost = {local_cost}")
            cost_history.append(local_cost)
    return [local_W, local_b, cost_history]

# %% [markdown]
# ## 단일 회귀예측

# %% [markdown]
# ### LinearRegression

# %%

linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
pred = linear_reg.predict(x_test)
mse_eval('LinearRegression', pred, y_test)

train_score = linear_reg.score(x_train, y_train)
test_score = linear_reg.score(x_test, y_test)
getscore('LinearRegression', train_score, test_score)




# %% [markdown]
# ### Ridge

# %%
ridge = Ridge(alpha=0.1)
ridge.fit(x_train, y_train)
pred = ridge.predict(x_test)
mse_eval('Ridge(alpha=0.1)', pred, y_test)

train_score = ridge.score(x_train, y_train)
test_score = ridge.score(x_test, y_test)
getscore('Ridge(alpha=0.1)', train_score, test_score)


# %% [markdown]
# ### Lasso

# %%
lasso = Lasso(alpha=0.01)
lasso.fit(x_train, y_train)
pred = lasso.predict(x_test)
mse_eval('Lasso(alpha=0.01)', pred, y_test)

train_score = lasso.score(x_train, y_train)
test_score = lasso.score(x_test, y_test)
getscore('Lasso(alpha=0.01)', train_score, test_score)

# %% [markdown]
# ### ElasticNetPermalink

# %%
elasticnet = ElasticNet(alpha=0.01, l1_ratio=0.8)
elasticnet.fit(x_train, y_train)
pred = elasticnet.predict(x_test)
mse_eval('ElasticNet(alpha=0.1, l1_ratio=0.8)', pred, y_test)  

train_score = elasticnet.score(x_train, y_train)
test_score = elasticnet.score(x_test, y_test)
getscore('ElasticNet(alpha=0.1, l1_ratio=0.8)', train_score, test_score)

# %% [markdown]
# ### Pipeline

# %%
elasticnet_pipeline = make_pipeline(
    StandardScaler(),
    ElasticNet(alpha=0.01, l1_ratio=0.8)
)
elasticnet_pipeline.fit(x_train, y_train)
elasticnet_pred = elasticnet_pipeline.predict(x_test)
mse_eval('Standard ElasticNet', elasticnet_pred, y_test)

train_score = elasticnet.score(x_train, y_train)
test_score = elasticnet.score(x_test, y_test)
getscore('Standard ElasticNet', train_score, test_score)

# %% [markdown]
# ### PolynomialFeaturesPermalink

# %%
from sklearn.pipeline import make_pipeline


# %%
poly_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    ElasticNet(alpha=0.1, l1_ratio=0.2)
)
poly_pipeline.fit(x_train, y_train)
poly_pred = poly_pipeline.predict(x_test)
mse_eval('Poly ElasticNet', poly_pred, y_test)

train_score = poly_pipeline.score(x_train, y_train)
test_score = poly_pipeline.score(x_test, y_test)
getscore('Poly ElasticNet', train_score, test_score)

# %% [markdown]
# ## 앙상블

# %% [markdown]
# ### 보팅회귀

# %%
from sklearn.ensemble import VotingRegressor

# %%
single_models = [
    ('linear_reg', linear_reg), 
    ('ridge', ridge), 
    ('lasso', lasso), 
    ('elasticnet_pipeline', elasticnet_pipeline), 
    ('poly_pipeline', poly_pipeline)
]
voting_regressor = VotingRegressor(single_models, n_jobs=-1)
voting_regressor.fit(x_train, y_train)
voting_pred = voting_regressor.predict(x_test)
mse_eval('Voting Ensemble', voting_pred, y_test)

train_score = voting_regressor.score(x_train, y_train)
test_score = voting_regressor.score(x_test, y_test)
getscore('Voting Ensemble', train_score, test_score)



# %% [markdown]
# ### 보팅분류

# %%
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# %%
models = [
    ('Logi', LogisticRegression()), 
    ('ridge', RidgeClassifier())
]
vc = VotingClassifier(models, voting='hard')
vc.fit(x_train, y_train)
vc_pred = vc.predict(x_test)
mse_eval('VotingClassifier-Hard', vc_pred, y_test)

train_score = vc.score(x_train, y_train)
test_score = vc.score(x_test, y_test)
getscore('VotingClassifier-Hard', train_score, test_score)

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
lr_clf = LogisticRegression()
knn_clf = KNeighborsClassifier(n_neighbors=8)
vc_soft = VotingClassifier(estimators=[('LR',lr_clf),('KNN',knn_clf)] , voting='soft')

vc_soft.fit(x_train, y_train)
vc_pred = vc_soft.predict(x_test)
mse_eval('VotingClassifier-Soft', vc_pred, y_test)

train_score = vc.score(x_train, y_train)
test_score = vc.score(x_test, y_test)
getscore('VotingClassifier-Soft', train_score, test_score)

# %% [markdown]
# ### 랜덤포레스트 

# %%
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_pred = rfr.predict(x_test)

mse_eval('RandomForest Ensemble', rfr_pred, y_test)

train_score = rfr.score(x_train, y_train)
test_score = rfr.score(x_test, y_test)
getscore('RandomForest Ensemble', train_score, test_score)

# %% [markdown]
# ### 랜덤포레스트 - 튜닝

# %%
rfr_t = RandomForestRegressor(random_state=42, n_estimators=1000, max_depth=7, max_features=0.9)
rfr_t.fit(x_train, y_train)
rfr_t_pred = rfr_t.predict(x_test)
mse_eval('RandomForest Ensemble w/ Tuning', rfr_t_pred, y_test)


train_score = rfr_t.score(x_train, y_train)
test_score = rfr_t.score(x_test, y_test)
getscore('RandomForest Ensemble w/ Tuning', train_score, test_score)

# %% [markdown]
# ### 부스팅 (AdaBoost / GradientBoost / LightGBM (LGBM) / XGBoost)

# %%
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBRegressor, XGBClassifier


# %% [markdown]
# ##### GradientBoosting - Regressor

# %%
gbr= GradientBoostingRegressor(random_state=42)
gbr.fit(x_train, y_train)
gbr_pred = gbr.predict(x_test)
mse_eval('GradientBoost Ensemble', gbr_pred, y_test)


train_score = gbr.score(x_train, y_train)
test_score = gbr.score(x_test, y_test)
getscore('GradientBoost Ensemble', train_score, test_score)

gbr= GradientBoostingRegressor(random_state=SEED, learning_rate=0.01)
gbr.fit(x_train, y_train)
gbr_pred = gbr.predict(x_test)
mse_eval('GradientBoosting(lr=0.01)', gbr_pred, y_test)


train_score = gbr.score(x_train, y_train)
test_score = gbr.score(x_test, y_test)
getscore('GradientBoosting(lr=0.01)', train_score, test_score)

gbr= GradientBoostingRegressor(random_state=SEED, learning_rate=0.01, n_estimators=1000)
gbr.fit(x_train, y_train)
gbr_pred = gbr.predict(x_test)
mse_eval('GradientBoosting(lr=0.01, est=1000)', gbr_pred, y_test)


train_score = gbr.score(x_train, y_train)
test_score = gbr.score(x_test, y_test)
getscore('GradientBoosting(lr=0.01, est=1000)', train_score, test_score)

gbr= GradientBoostingRegressor(random_state=SEED, learning_rate=0.01, n_estimators=1000, subsample=0.8)
gbr.fit(x_train, y_train)
gbr_pred = gbr.predict(x_test)
mse_eval('GradientBoosting(lr=0.01, est=1000, subsample=0.8)', gbr_pred, y_test)


train_score = gbr.score(x_train, y_train)
test_score = gbr.score(x_test, y_test)
getscore('GradientBoosting(lr=0.01, est=1000, subsample=0.8)', train_score, test_score)

# %% [markdown]
# ##### XGBoost

# %%
xgb = XGBRegressor(random_state=SEED)
xgb.fit(x_train, y_train)
xgb_pred = gbr.predict(x_test)
mse_eval('XGBoost', xgb_pred, y_test)


train_score = xgb.score(x_train, y_train)
test_score = xgb.score(x_test, y_test)
getscore('XGBoost', train_score, test_score)

xgb = XGBRegressor(random_state=42, learning_rate=0.01, n_estimators=1000, subsample=0.8, max_features=0.8, max_depth=7)
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
mse_eval('XGBoost w/ Tuning', xgb_pred, y_test)

train_score = xgb.score(x_train, y_train)
test_score = xgb.score(x_test, y_test)
getscore('XGBoost w/ Tuning', train_score, test_score)



# %% [markdown]
# ##### LGBM-Regressor

# %%
lgbm = LGBMRegressor(random_state=SEED)
lgbm.fit(x_train, y_train)
lgbm_pred = lgbm.predict(x_test)
mse_eval('LGBM', lgbm_pred, y_test)

train_score = lgbm.score(x_train, y_train)
test_score = lgbm.score(x_test, y_test)
getscore('LGBM', train_score, test_score)

lgbm = LGBMRegressor(random_state=SEED, learning_rate=0.01, n_estimators=1500, colsample_bytree=0.9, num_leaves=15, subsample=0.8)
lgbm.fit(x_train, y_train)
lgbm_pred = lgbm.predict(x_test)
mse_eval('LGBM w/ Tuning', lgbm_pred, y_test)

train_score = lgbm.score(x_train, y_train)
test_score = lgbm.score(x_test, y_test)
getscore('LGBM w/ Tuning', train_score, test_score)





# %% [markdown]
# ##### Stacking

# %%
import sklearn
sklearn.__version__
from sklearn.ensemble import StackingRegressor

stack_models = [
    ('elasticnet', poly_pipeline), 
    ('randomforest', rfr), 
    ('gbr', gbr),
    ('lgbm', lgbm),
]

# %%
stack_reg = StackingRegressor(stack_models, final_estimator=xgb, n_jobs=-1)
stack_reg.fit(x_train, y_train)
stack_pred = stack_reg.predict(x_test)
mse_eval('Stacking Ensemble', stack_pred, y_test)

train_score = stack_reg.score(x_train, y_train)
test_score = stack_reg.score(x_test, y_test)
getscore('Stacking Ensemble', train_score, test_score)



# %%


# %% [markdown]
# ##### Weighted Blending

# %%
final_outputs = {
    'elasticnet': poly_pred, 
    'randomforest': rfr_pred, 
    'gbr': gbr_pred,
    'xgb': xgb_pred, 
    'lgbm': lgbm_pred,
    'stacking': stack_pred,
}

# %%
final_prediction=\
final_outputs['elasticnet'] * 0.1\
+final_outputs['randomforest'] * 0.1\
+final_outputs['gbr'] * 0.2\
+final_outputs['xgb'] * 0.25\
+final_outputs['lgbm'] * 0.15\
+final_outputs['stacking'] * 0.2

# %%
mse_eval('Weighted Blending', final_prediction, y_test)
# weighted_blending = StackingRegressor(final_outputs, final_estimator=xgb, n_jobs=-1)
# weighted_blending.fit(x_train, y_train)
# mse_eval('Weighted Blending2', final_prediction, y_test)

# train_score = weighted_blending.score(x_train, y_train)
# test_score = weighted_blending.score(x_test, y_test)
# getscore('Weighted Blending2', train_score, test_score)

# %% [markdown]
# ### 검증과 튜닝

# %% [markdown]
# ##### K-Fold Cross Validation

# %%
from sklearn.model_selection import KFold
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=SEED)

# %%
from sklearn.model_selection import train_test_split

X=np.array(df.iloc[:,:-1])
Y=np.array(df["gmm_cluster"])
X.shape, Y.shape

# x_train, x_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.3, random_state=42
# )


# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# %%
lgbm_fold = LGBMRegressor(random_state=SEED)

# %%
i = 1
total_error = 0
for train_index, test_index in kfold.split(X):
    x_train_fold, x_valid_fold = X[train_index], X[test_index]
    y_train_fold, y_valid_fold = Y[train_index], Y[test_index]
    lgbm_pred_fold = lgbm_fold.fit(x_train_fold, y_train_fold).predict(x_valid_fold)
    error = mean_squared_error(lgbm_pred_fold, y_valid_fold)
    print('Fold = {}, prediction score = {:.2f}'.format(i, error))
    total_error += error
    i+=1
print('---'*10)
print('Average Error: %s' % (total_error / n_splits))

# %% [markdown]
# ### Hyperparameter 튜닝

# %% [markdown]
# #### RandomizedSearchCV

# %%
params = {
    'n_estimators': [200, 500, 1000, 2000], 
    'learning_rate': [0.1, 0.05, 0.01], 
    'max_depth': [6, 7, 8], 
    'colsample_bytree': [0.8, 0.9, 1.0], 
    'subsample': [0.8, 0.9, 1.0],
}

from sklearn.model_selection import RandomizedSearchCV


# %%
clf = RandomizedSearchCV(LGBMRegressor(), params, random_state=42, cv=3, n_iter=25, scoring='neg_mean_squared_error')
clf.fit(x_train, y_train)

# %%
clf.best_score_, clf.best_params_

# %%
lgbm_best = LGBMRegressor(n_estimators=2000, subsample=0.8, max_depth=7, learning_rate=0.01, colsample_bytree=0.8)
lgbm_best.fit(x_train, y_train)
lgbm_best_pred = lgbm_best.predict(x_test)
mse_eval('RandomSearch LGBM', lgbm_best_pred, y_test)

train_score = lgbm_best.score(x_train, y_train)
test_score = lgbm_best.score(x_test, y_test)
getscore('RandomSearch LGBM', train_score, test_score)

# %% [markdown]
# #### GridSearchCVPermalink

# %%
params = {
    'n_estimators': [500, 1000], 
    'learning_rate': [0.1, 0.05, 0.01], 
    'max_depth': [7, 8], 
    'colsample_bytree': [0.8, 0.9], 
    'subsample': [0.8, 0.9,],
}

from sklearn.model_selection import GridSearchCV

# LGBMRegressor 초기화
lgbm = LGBMRegressor()

# GridSearchCV 설정
grid_search = GridSearchCV(
    estimator=lgbm, 
    param_grid=params, 
    scoring='neg_mean_squared_error', 
    cv=5, 
    n_jobs=-1, 
    verbose=1
)



# %% [markdown]
# grid_search = GridSearchCV(LGBMRegressor(), params, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
# grid_search.fit(x_train, y_train)
# 
# abs(grid_search.best_score_), grid_search.best_params_
#  

# %%

# GridSearchCV 실행하여 최적의 파라미터 찾기
grid_search.fit(x_train, y_train)  # fit() 메서드 실행

# 최적의 파라미터 확인
print("Best parameters found: ", grid_search.best_params_)

# 최적의 파라미터로 모델 학습
lgbm_best = LGBMRegressor(**grid_search.best_params_)
lgbm_best.fit(x_train, y_train)

# 예측
lgbm_best_pred = lgbm_best.predict(x_test)

# 평가 함수 사용 (mse_eval과 getscore가 정의되어 있다고 가정)
mse_eval('GridSearch LGBM', lgbm_best_pred, y_test)

train_score = lgbm_best.score(x_train, y_train)
test_score = lgbm_best.score(x_test, y_test)
getscore('GridSearch LGBM', train_score, test_score)

# %%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# %%
# 변수 구분
X = df.iloc[:,:-1]
y = df["gmm_cluster"]

# 데이터셋을 학습용과 테스트용으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 학습 - Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# 모델 학습 - Random Forest
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# %%
# 예측
y_pred_tree = decision_tree.predict(X_test)
y_pred_forest = random_forest.predict(X_test)

# %%
# 모델 평가 - 정확도, 혼동 행렬, ROC 곡선
accuracy_tree = accuracy_score(y_test, y_pred_tree)
accuracy_forest = accuracy_score(y_test, y_pred_forest)

conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
conf_matrix_forest = confusion_matrix(y_test, y_pred_forest)


# %%
accuracy_tree, accuracy_forest, conf_matrix_tree, conf_matrix_forest

# %%
# 시각화
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(accuracy_tree, conf_matrix_tree, label=f"Decision Tree (AUC = {accuracy_forest:.2f})")
plt.plot(accuracy_forest, conf_matrix_forest, label=f"Random Forest (AUC = {accuracy_forest:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_tree, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# 모델 평가 결과 출력
print("Decision Tree Accuracy:", accuracy_tree)
print("Random Forest Accuracy:", accuracy_forest)
print("\nClassification Report - Decision Tree:\n", classification_report(y_test, y_pred_tree))
print("\nClassification Report - Random Forest:\n", classification_report(y_test, y_pred_forest))

# %%
# 1. Random Forest 하이퍼 파라미터 튜닝
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(X_train, y_train)

# %%
# 최적의 하이퍼파라미터 출력
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# 최적의 모델로 예측 및 평가
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
print("Best Random Forest Accuracy:", accuracy_best_rf)

# %%
# 2. 피처 중요도 도출 및 시각화
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc

plt.rcParams["font.family"] = "Malgun Gothic" 


feature_importances = best_rf.feature_importances_

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=X.columns)
plt.title("Feature Importances from Best Random Forest Model")
plt.show()

# %%
X

# %% [markdown]
# ### 군집분석

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd

#case n_components =2 2차원으로 축소
pca = PCA(n_components=2) #feature를 두개로 축소하라
data_scaled = StandardScaler().fit_transform(df) # scaling
pca_data = pca.fit_transform(data_scaled)

# 시각화
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
%matplotlib inline

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['gmm_cluster'])


#case n_components =0.9 3차원을 유지하되 분산을 0.99로
pca = PCA(n_components=0.99)
pca_data = pca.fit_transform(data_scaled)

from mpl_toolkits.mplot3d import Axes3D
import numpy as np 

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d') # Axe3D object

sample_size = 50
ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], alpha=0.6, c=df['gmm_cluster'])
plt.savefig('./tmp.svg')
plt.title("ax.plot")
plt.show()



# %%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
cluster_data = kmeans.fit_transform(df)
cluster_data[:5]
kmeans.labels_
sns.countplot(df['gmm_cluster']) 

#param 변경해서 model 수정
kmeans = KMeans(n_clusters=3, max_iter=500)
cluster_data = kmeans.fit_transform(df)
sns.countplot(kmeans.labels_)

# %%
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=2) #eps와 minsample을 통해 적절하게 군집화를 알아서 실행
dbscan_data = dbscan.fit_predict(df)
dbscan_data

# %%
from sklearn.metrics import silhouette_samples, silhouette_score
score = silhouette_score(data_scaled, kmeans.labels_)

samples = silhouette_samples(data_scaled, kmeans.labels_)
samples[:5]

# %%
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

# # Generating the sample data from make_blobs
# # This particular setting has one distinct cluster and 3 clusters placed close
# # together.
# X, y = make_blobs(
#     n_samples=500,
#     n_features=2,
#     centers=4,
#     cluster_std=1,
#     center_box=(-10.0, 10.0),
#     shuffle=True,
#     random_state=1,
# )  # For reproducibility

X = df.iloc[:, :-1].values  # DataFrame을 numpy 배열로 변환
y = df["gmm_cluster"]

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()

# %%
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

# 데이터 준비 (예시로 df에서 데이터 불러옴)
X = df.iloc[:, :-1] # 독립 변수
y = df["신차대비가격분류"]  # 종속 변수

# DBSCAN 파라미터 설정
eps_values = [0.5, 1.0, 1.5, 2.0]  # eps 값을 증가시켜서 시도
min_samples = 5  # 각 군집의 최소 샘플 수

for eps in eps_values:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + 10])

    # DBSCAN 클러스터링 수행
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(X)

    # 군집 수 계산 (잡음으로 인한 -1 라벨 제외)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    if n_clusters <= 1:
        print(f"For eps = {eps}, the algorithm identified {n_clusters} cluster(s). Skipping silhouette analysis.")
        plt.close(fig)  # 플롯을 그리지 않고 닫습니다.
        continue

    # 실루엣 점수 계산
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"For eps = {eps}, the average silhouette_score is : {silhouette_avg}")

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    # 잡음 (노이즈) 포인트를 별도로 표시
    ax2.scatter(
        X[cluster_labels == -1, 0],
        X[cluster_labels == -1, 1],
        marker="x",
        c="black",
        alpha=1,
        s=100,
        edgecolor="k",
        label="Noise"
    )

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        f"Silhouette analysis for DBSCAN clustering with eps = {eps}",
        fontsize=14,
        fontweight="bold",
    )

    plt.show()



# %%



