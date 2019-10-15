import numpy as np
import pandas as pd
titanic_df = pd.read_csv('train.csv')
titanic_df['Age_0'] = 0
titanic_df.head(3)
titanic_df.drop('Age_0', axis=1, inplace=True)
indexes = titanic_df.index
print(indexes)
print(indexes.values)
series_fare = titanic_df['Fare']
print(series_fare.max())
print(series_fare.min())
print(series_fare.head())
titanic_reset_df = titanic_df.reset_index(inplace=False)
titanic_reset_df.head(3)
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
new_value_counts = value_counts.reset_index(inplace=False)
print(new_value_counts)
titanic_df[titanic_df['Pclass']==3].head(3)
titanic_df.iloc[1,]

data = {'Name':['chulmin', 'Eunkyung', 'Jonwoong', 'Soobeom'],
        'Year' : [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']}
data_df = pd.DataFrame(data, index=['one', 'two', 'three', 'four'])
data_df
data_df.reset_index(drop=True, inplace=True)
data_df.loc[:3,'Name':'Year']
data_df.iloc[:3,:2]
#정리하자면, loc은 명칭기반인덱싱이다. 그래서 그냥 하나씩 뽑는 인덱싱에서는 숫자를 인식하지만,
#슬라이싱을 할 때 : 행-숫자 인식 가능, 열-무조건 문자열. 근데 마지막 입력값도 포함해서 출력
#iloc은 위치기반인덱싱이므로 슬라이싱에서도 숫자 인식 가능. 얘는 원래 슬라이싱처럼 마지막 -1해서 출력
titanic_df[titanic_df['Age']>60][['Age', 'Name']]
titanic_df.loc[titanic_df['Age']>60, ['Name','Age']]

##lambda를 이용하는 식
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x:'Child' if x <= 15 else('Adult' if x <= 60 else 'Elderly'))
titanic_df.groupby('Age_cat').count()

def get_category(age):
        cat = ''
        if age <= 5: cat = 'Baby'
        elif age <= 12: cat = 'Child'
        elif age <= 18: cat = 'Teenage'
        elif age <= 25: cat = 'Student'
        elif age <= 35: cat = 'Young Adult'
        elif age <= 60: cat = 'Adult'
        else: cat = 'Elderly'

        return cat

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
print(titanic_df['Age_cat'].head(10))

import sklearn
print(sklearn.__version__)
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
iris = load_iris()
iris_data = iris.data
iris_label = iris.target
print(iris_data)
iris_label
iris_df = pd.DataFrame(data = iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df.head())
print(iris.feature_names)

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label,
                                                   test_size=0.2, random_state=11)
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

iris = load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target
dt_clf.fit(train_data, train_label)
pred = dt_clf.predict(train_data)
print(accuracy_score(train_label, pred))

##학습,테스트 데이터 분리
dt_clf = DecisionTreeClassifier()
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, \
                                                    test_size=0.3, random_state = 121)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print('예측정확도 : {0:.4f}'.format(accuracy_score(y_test, pred)))
#########################################
iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# KFold를 가지고 교차검증
kfold = KFold(n_splits=1)
cv_accuracy = []
print('붓꽃 데이터 세트 크기:', features.shape[0])
n_iter = 0

#KFold 객체의 split()를 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 array로 변환
for train_index, test_index in kfold.split(features):
        #kfold.split()으로 변한된 인덱스를 이용해 학습용, 검증용 테스트 데이터를 추출
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = label[train_index], label[test_index]
        #학습 및 예측
        dt_clf.fit(X_train, y_train)
        pred = dt_clf.predict(X_test)
        n_iter += 1
        #반복 시마다 정확도 측정
        accuracy = np.round(accuracy_score(y_test, pred), 4)
        train_size = X_train.shape[0]
        test_size = X_test.shape[0]
        # print('\n#{0} 교차 검증 정확도 :{1}, 학습데이터 크기: {2}, 검증데이터 크기: {3}'
        #       .format(n_iter, accuracy, train_size, test_size))
        # print('#{0} 검증 세트 인덱스:{1}'.format(n_iter, test_index))
        cv_accuracy.append(accuracy)

#개별 iteration별 정확도를 합하여 평균 정확도 계산
# print('\n## 평균 검증 정확도:', np.mean(cv_accuracy))

##stratified K fold
import pandas as pd
iris = load_iris()
iris_df = pd.DataFrame(data = iris.data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df['label'].value_counts()

kfold = KFold(n_splits=3)
n_iter = 0
for train_index, test_index in kfold.split(iris_df):
        n_iter += 1
        label_train = iris_df['label'].iloc[train_index]
        label_test = iris_df['label'].iloc[test_index]
        print('##교차검증:{0}'.format(n_iter))