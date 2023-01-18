import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
def warn(*args,**kwargs):
    pass
warnings.warn = warn

Insurance_Claim = pd.read_csv("C:\\Information_Science\\My_projects\\train.csv")


X = Insurance_Claim.iloc[:,[2,21]].values
y = Insurance_Claim.iloc[:, 30].values
onehotencoder = OneHotEncoder(sparse=False)
Z = onehotencoder.fit_transform(X[:,[0]])
X = np.hstack((X[:,:0],Z)).astype('int')
onehotencoder = OneHotEncoder(sparse=False)
Z = onehotencoder.fit_transform(X[:,[1]])
X = np.hstack((X[:,:1],Z)).astype('int')
X = X[:, :-1]

colormap = plt.cm.viridis
plt.figure(figsize=(30,30))
plt.title('Pearson correlation of attributes',y=1.05,size=19)
sns.heatmap(Insurance_Claim.corr(),linewidths=0.1,vmax=1.0,
            square=True,cmap=colormap,linecolor='white',annot=True)
print(sns.heatmap(Insurance_Claim.corr()))
print(plt.show())
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svc = SVC()
svc.fit(X_train,y_train)
pred = svc.predict(X_test)
print("Score of Age and Gear type variables:",svc.score(X_train,y_train))
print(svc.score(X_test,y_test))

from sklearn import metrics
print(metrics.confusion_matrix(y_test,pred))

from imblearn.over_sampling import SMOTE
over_sample = SMOTE()
x_smote,y_smote = over_sample.fit_resample(X_train,y_train)
print(sns.countplot(y_smote))
print(sns.countplot(pred))


