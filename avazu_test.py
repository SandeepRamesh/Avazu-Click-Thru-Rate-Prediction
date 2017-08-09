# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:27:42 2017

@author: sande
"""
#Importing Libraries
import csv, sqlite3
import odo
import pandas as pd
import seaborn as sns
import blaze as bz
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


#using Odo to convert csv data to sqlite db
csv_path = 'train.csv'
bz.odo(csv_path, 'sqlite:///data.db::data')

#using pandas to read 300,000 sample rows created from sqlite db out of 8 million rows
data_df=pd.read_csv('train_300k.csv','r',delimiter=',')
#data_df=data_df.drop(['id'],1)
data_df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 23 columns):
hour                100000 non-null int64
C1                  100000 non-null int64
banner_pos          100000 non-null int64
site_id             100000 non-null object
site_domain         100000 non-null object
site_category       100000 non-null object
app_id              100000 non-null object
app_domain          100000 non-null object
app_category        100000 non-null object
device_id           100000 non-null object
device_ip           100000 non-null object
device_model        100000 non-null object
device_type         100000 non-null int64
device_conn_type    100000 non-null int64
C14                 100000 non-null int64
C15                 100000 non-null int64
C16                 100000 non-null int64
C17                 100000 non-null int64
C18                 100000 non-null int64
C19                 100000 non-null int64
C20                 100000 non-null int64
C21                 100000 non-null int64
click               100000 non-null int64
dtypes: int64(14), object(9)
memory usage: 17.5+ MB
'''
data_df.describe()
data_df.describe(include=['object'])

#Data Exploration of each feature

#hour feature to str
data_df.hour=data_df.hour.astype('str')
data_df['hour']=data_df.hour.str[6:8]
sns.countplot(x='hour',data=data_df,hue=data_df.click)
data_df.hour=data_df.hour.astype('int')
#probability to click a ad is more after 2PM

#feature C1
data_df.C1.value_counts()
#Since it contains in the form of categories , this is mostly some categorical feature

sns.factorplot(x='C1',y='click',data=data_df)
sns.countplot(x='C1',hue='click',data=data_df)
#1002 and 1005 have mean click probability above 0.5 and the rest below 0.5
data_df['C1']=data_df.C1.replace(to_replace=[1002,1005],value=1)
data_df['C1']=data_df.C1.replace(to_replace=[1010,1012,1007,1008,1001],value=0)
#converting to category datatype
data_df.C1=data_df.C1.astype('category')

#Feature banner_pos
sns.countplot(x='banner_pos',hue='click',data=data_df)
sns.factorplot(x='banner_pos',y='click',data=data_df)
data_df.banner_pos=data_df.banner_pos.astype('category')

#Feature Site_Id
data_df.site_id=data_df.site_id.map(lambda x:x[0])
data_df.groupby('site_id')['click'].mean().sort_values()
data_df['site_id']=data_df.site_id.replace(to_replace=['b','c','a','2','7','8','f','6','4'],value=0)
data_df['site_id']=data_df.site_id.replace(to_replace=['0','9'],value=1)
data_df['site_id']=data_df.site_id.replace(to_replace=['1','3'],value=2)
data_df['site_id']=data_df.site_id.replace(to_replace=['e','5'],value=3)
data_df['site_id']=data_df.site_id.replace(to_replace=['d'],value=4)
sns.factorplot(x='site_id',y='click',data=data_df)
data_df.site_id=data_df.site_id.astype('category')

#Feature Site_domain
data_df.site_domain=data_df.site_domain.map(lambda x:x[0])
data_df.groupby('site_domain')['click'].mean().sort_values()
data_df['site_domain']=data_df.site_domain.replace(to_replace=['d','3','8','5','a','6','2','b','c'],value=0)
data_df['site_domain']=data_df.site_domain.replace(to_replace=['0','4','e','1'],value=1)
data_df['site_domain']=data_df.site_domain.replace(to_replace=['f','9'],value=2)
data_df['site_domain']=data_df.site_domain.replace(to_replace=['7'],value=3)
sns.factorplot(x='site_domain',y='click',data=data_df)
data_df.site_domain=data_df.site_domain.astype('category')

#Feature Site Category
data_df.site_category=data_df.site_category.map(lambda x:x[0])
data_df.groupby('site_category')['click'].mean().sort_values()
data_df['site_category']=data_df.site_category.replace(to_replace=['b','0','7','4','c','5','f','a'],value=0)
data_df['site_category']=data_df.site_category.replace(to_replace=['8','e'],value=1)
data_df['site_category']=data_df.site_category.replace(to_replace=['2'],value=2)
data_df['site_category']=data_df.site_category.replace(to_replace=['3','d'],value=3)
sns.factorplot(x='site_category',y='click',data=data_df)
data_df.site_category=data_df.site_category.astype('category')

#Feature App ID
data_df.app_id=data_df.app_id.map(lambda x:x[0])
data_df.groupby('app_id')['click'].mean().sort_values()
data_df['app_id']=data_df.app_id.replace(to_replace=['c','f','4','3','1','5','8','b','d','0','2','6'],value=0)
data_df['app_id']=data_df.app_id.replace(to_replace=['e','9'],value=1)
data_df['app_id']=data_df.app_id.replace(to_replace=['7','a'],value=2)
sns.factorplot(x='app_id',y='click',data=data_df)
data_df.app_id=data_df.app_id.astype('category')

#Feature app_domain
data_df.app_domain=data_df.app_domain.map(lambda x:x[0])
data_df.groupby('app_domain')['click'].mean().sort_values()
data_df['app_domain']=data_df.app_domain.replace(to_replace=['c','1','f','8','a','d','e','3','2','0','4','9','5'],value=0)
data_df['app_domain']=data_df.app_domain.replace(to_replace=['7'],value=1)
data_df['app_domain']=data_df.app_domain.replace(to_replace=['6'],value=2)
data_df['app_domain']=data_df.app_domain.replace(to_replace=['b'],value=3)
sns.factorplot(x='app_domain',y='click',data=data_df)
data_df.app_domain=data_df.app_domain.astype('category')

#Feature App_Category
data_df.app_category=data_df.app_category.map(lambda x:x[0])
data_df.groupby('app_category')['click'].mean().sort_values()
data_df['app_category']=data_df.app_category.replace(to_replace=['2','7','d','8','a','c','4'],value=0)
data_df['app_category']=data_df.app_category.replace(to_replace=['0','f','1'],value=1)
sns.factorplot(x='app_category',y='click',data=data_df)
data_df.app_category=data_df.app_category.astype('category')

#Feature Device ID
data_df.device_id=data_df.device_id.map(lambda x:x[0])
data_df.groupby('device_id')['click'].mean().sort_values()
data_df['device_id']=data_df.device_id.replace(to_replace=['f','8','e','2','6','5','7','9','b','4','3','0','1','d'],value=0)
data_df['device_id']=data_df.device_id.replace(to_replace=['a'],value=1)
data_df['device_id']=data_df.device_id.replace(to_replace=['c'],value=2)
sns.factorplot(x='device_id',y='click',data=data_df)
data_df.device_id=data_df.device_id.astype('category')

#Feature Device IP
data_df.device_ip=data_df.device_ip.map(lambda x:x[0])
data_df.groupby('device_ip')['click'].mean().sort_values()
data_df['device_ip']=data_df.device_ip.replace(to_replace=['e','c','3','2','4','D','d','5','9'],value=0)
data_df['device_ip']=data_df.device_ip.replace(to_replace=['0','b','f','a','6','7'],value=1)
data_df['device_ip']=data_df.device_ip.replace(to_replace=['1','8'],value=2)
sns.factorplot(x='device_ip',y='click',data=data_df)
data_df.device_ip=data_df.device_ip.astype('category')

#Feature Device Model
data_df.device_model=data_df.device_model.map(lambda x:x[0])
data_df.groupby('device_model')['click'].mean().sort_values()
data_df['device_model']=data_df.device_model.replace(to_replace=['9','2','5','3','8','b','a','f','e'],value=0)
data_df['device_model']=data_df.device_model.replace(to_replace=['6','d','0','7','1','4','c'],value=1)
sns.factorplot(x='device_model',y='click',data=data_df)
data_df.device_model=data_df.device_model.astype('category')

#Feature Device Type
data_df.groupby('device_type')['click'].mean().plot(kind='bar')
data_df.device_type=data_df.device_type.astype('category')
sns.factorplot(x='device_type',y='click',data=data_df)

#Feature Device conn Type
sns.factorplot(x='device_conn_type',y='click',data=data_df)
data_df.device_conn_type=data_df.device_conn_type.astype('category')

#Feature C15
sns.factorplot(x='C15',y='click',data=data_df)
data_df['C15']=data_df.C15.replace(to_replace=[728,216,320],value=0)
data_df['C15']=data_df.C15.replace(to_replace=[300,480,768,1024],value=1)
data_df.C15=data_df.C15.astype('category')

#Feature C16
data_df.groupby('C16')['click'].mean().sort_values()
data_df['C16']=data_df.C16.replace(to_replace=[90,36,50],value=0)
data_df['C16']=data_df.C16.replace(to_replace=[480,320,1024,250,768],value=1)
data_df.C16=data_df.C16.astype('category')
sns.factorplot(x='C16',y='click',data=data_df)

#Feature C18 to category
data_df.C18=data_df.C18.astype('category')
sns.factorplot(x='C18',y='click',data=data_df)

#Scaling certain features to mean 0 and std dev 1
scaler = StandardScaler()
for col in ['C14','C17','C19','C20','C21']:
    data_df[col]=scaler.fit_transform(data_df[col])
    
    
    
x=data_df.drop(['id','click'],1)
y=data_df['click']

#using train test split to get 80% training and 20% test data
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)

#Recursive feature elimination to select the optimal number of features
regressor = RandomForestClassifier(n_estimators=200)
rfecv = RFECV(estimator=regressor, step=1, cv=10)
rfecv.fit(x_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#new dataframe with features and their rankings
ranking=pd.DataFrame({'Features': x_train.columns})
ranking['rank'] = rfecv.ranking_
ranking.sort_values('rank',inplace=True)
ranking.to_csv('Ranking.csv',index=False)

#Using best 16 features according to the plot above
newfeatures=ranking.Features[:16]
new_xtrain=x_train[newfeatures]
new_ytrain=y_train
new_xtest=x_test[newfeatures]
new_ytest=y_test


#List of classifiers to be used for Model selection and validation
names = ["Extra Trees", "Random Forest", "KNeighbors","Logistic",
         "Naive Bayes", "Decision Tree","AdaBoost"]
classifiers = [
    ExtraTreesClassifier(n_estimators=200,criterion = 'entropy'),
    RandomForestClassifier(n_estimators=200,criterion = 'entropy'),
    KNeighborsClassifier(),
    LogisticRegression(),
    GaussianNB(),
    DecisionTreeClassifier(criterion='entropy'),
    AdaBoostClassifier(n_estimators=200)
]


#Model training and to validate using F1 score, Accuracy, AUC score 
#Plotting of ROC Curve to choose the best classifier
i=0
f1_results=[]
acc_results=[]
auc_results=[]
for classifier in classifiers:
    print(names[i])
    classifier.fit(new_xtrain, new_ytrain)
    y_pred = classifier.predict(new_xtest)
    y_pred_prob = classifier.predict_proba(new_xtest)[:, 1]
    f1score=f1_score(new_ytest,y_pred)
    accuracy=accuracy_score(new_ytest,y_pred)
    auc_score=roc_auc_score(new_ytest, y_pred_prob)
    print("F1 Score:",f1score)
    print("Accuracy Score:",accuracy)
    print("ROC AUC Score:",auc_score) 
    f1_results.append(f1score)
    acc_results.append(accuracy)
    auc_results.append(auc_score)
    
    #To plot ROC Curve to select the best performing classifier
    fpr, tpr, thresholds = roc_curve(new_ytest, y_pred_prob)
    plt.plot(fpr, tpr,label=names[i])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for all classifiers')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc='best')
    plt.grid(True)
    i+=1

#Magnified ROC Curve
i=0
f1_results=[]
acc_results=[]
auc_results=[]
for classifier in classifiers:
    print(names[i])
    classifier.fit(new_xtrain, new_ytrain)
    y_pred = classifier.predict(new_xtest)
    y_pred_prob = classifier.predict_proba(new_xtest)[:, 1]
    f1score=f1_score(new_ytest,y_pred)
    accuracy=accuracy_score(new_ytest,y_pred)
    auc_score=roc_auc_score(new_ytest, y_pred_prob)
    print("F1 Score:",f1score)
    print("Accuracy Score:",accuracy)
    print("ROC AUC Score:",auc_score) 
    f1_results.append(f1score)
    acc_results.append(accuracy)
    auc_results.append(auc_score)
    
    #To plot ROC Curve to select the best performing classifier
    fpr, tpr, thresholds = roc_curve(new_ytest, y_pred_prob)
    plt.plot(fpr, tpr,label=names[i])
    plt.xlim([0.2, 0.8])
    plt.ylim([0.2, 0.8])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for all classifiers(Magnified)')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc='best')
    plt.grid(True)
    i+=1
