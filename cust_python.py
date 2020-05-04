#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')


# In[2]:


train_df = pd.read_csv("C:/Users/kanch/OneDrive/Desktop/project/Project_3/train.csv")


# In[3]:


test_df = pd.read_csv("C:/Users/kanch/OneDrive/Desktop/project/Project_3/test.csv")


# In[4]:


os.getcwd()


# In[5]:


##################Exploring the data####################
train_df.shape


# In[6]:


test_df.shape


# In[7]:


#Both train and test data have 200,000 rows and train have 202 and test have 201 variables
#lets look into train and test dataset
train_df.head()


# In[8]:


test_df.head()


# In[9]:


############Missing value Analysis###################
def miss_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[10]:


miss_data(train_df)


# In[11]:


miss_data(test_df)


# In[12]:


#There are no missing data in train and test datasets. Let's check the numerical values in train and test dataset.
train_df.describe()


# In[13]:


test_df.describe()


# In[14]:


#here we can observe the following
#standard deviation is relatively large for both train and test variable data
#min, max, mean, sdt values for train and test data looks little close
#mean values are distributed over a large range.


# In[15]:


#The number of values in train and test set is the same.
#Let's plot the scatter plot for train and test set for few of the features.
def plot_feature_scatter(df1, df2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(4,4,figsize=(14,14))

    for feature in features:
        i += 1
        plt.subplot(4,4,i)
        plt.scatter(df1[feature], df2[feature], marker='+')
        plt.xlabel(feature, fontsize=9)
    plt.show();


# In[16]:


#let's plot 5% of the data. x axis shows train values and y axis shows the train values.
features = ['var_0', 'var_1','var_2','var_3', 'var_4', 'var_5', 'var_6', 'var_7', 
           'var_8', 'var_9', 'var_10','var_11','var_12', 'var_13', 'var_14', 'var_15', 
           ]
plot_feature_scatter(train_df[::20],test_df[::20], features)


# In[18]:


#distribution of target value in train dataset
sns.countplot(train_df['target'], palette='Set3')


# In[19]:


print("There are {}% target values with 1".format(100 * train_df["target"].value_counts()[1]/train_df.shape[0]))


# In[20]:


#checking outliers using Chauvenet's criterion
def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Lenght of incoming array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)                # Area normal dist.    
    return prob < criterion       # Use boolean array outside this function


# In[21]:


numerical_features=train_df.columns[2:]


# In[22]:


from scipy.special import erfc
train_outliers = dict()
for col in [col for col in numerical_features]:
    train_outliers[col] = train_df[chauvenet(train_df[col].values)].shape[0]
train_outliers = pd.Series(train_outliers)

train_outliers.sort_values().plot(figsize=(14, 40), kind='barh').set_xlabel('Number of outliers');


# In[23]:


print('Total number of outliers in training set: {} ({:.2f}%)'.format(sum(train_outliers.values), (sum(train_outliers.values) / train_df.shape[0]) * 100))


# In[24]:


#outliers in each variable in test data 
test_outliers = dict()
for col in [col for col in numerical_features]:
    test_outliers[col] = test_df[chauvenet(test_df[col].values)].shape[0]
test_outliers = pd.Series(test_outliers)

test_outliers.sort_values().plot(figsize=(14, 40), kind='barh').set_xlabel('Number of outliers');


# In[25]:


print('Total number of outliers in testing set: {} ({:.2f}%)'.format(sum(test_outliers.values), (sum(test_outliers.values) / test_df.shape[0]) * 100))


# In[26]:


#remove these outliers in train and test data
for col in numerical_features:
    train_df=train_df.loc[(~chauvenet(train_df[col].values))]
for col in numerical_features:
    test_df=test_df.loc[(~chauvenet(test_df[col].values))]


# In[27]:


#Let's see the density plot of variables in train dataset.
#We represent with different colors the distribution for values with target value 0 and 1.
def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


# In[28]:


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[2:102]
plot_feature_distribution(t0, t1, '0', '1', features)


# In[30]:


#Let's see the distribution of the same features in parallel in train and test datasets.
features = train_df.columns.values[2:102]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)


# In[31]:


#The train and test data are well balanced with respect with distribution of the numeric variables.


# In[32]:


#Distribution of mean and std
plt.figure(figsize=(16,6))
features = train_df.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train_df[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(test_df[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[33]:


#Let's check the distribution of the mean values per columns in the train and test set.

plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train_df[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')
sns.distplot(test_df[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[34]:


#Let's show the distribution of standard deviation of values per row for train and test datasets.

plt.figure(figsize=(16,6))
plt.title("Distribution of std values per row in the train and test set")
sns.distplot(train_df[features].std(axis=1),color="black", kde=True,bins=120, label='train')
sns.distplot(test_df[features].std(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend();plt.show()


# In[35]:


#Let's check the distribution of the standard deviation of values per columns in the train and test datasets.

plt.figure(figsize=(16,6))
plt.title("Distribution of std values per column in the train and test set")
sns.distplot(train_df[features].std(axis=0),color="blue",kde=True,bins=120, label='train')
sns.distplot(test_df[features].std(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend(); plt.show()


# In[36]:


#Distribution of skew and kurtosis
#distribution of skewness calculated per rows in train

plt.figure(figsize=(16,6))
plt.title("Distribution of skew per row in the train and test set")
sns.distplot(train_df[features].skew(axis=1),color="red", kde=True,bins=120, label='train')
sns.distplot(test_df[features].skew(axis=1),color="orange", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[37]:


#distribution of skewness calculated per columns in train and test set.

plt.figure(figsize=(16,6))
plt.title("Distribution of skew per column in the train and test set")
sns.distplot(train_df[features].skew(axis=0),color="magenta", kde=True,bins=120, label='train')
sns.distplot(test_df[features].skew(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[38]:


#Features correlation
#correlations between the features in train set.
#The following table shows the first 10 the least correlated features.

correlations = train_df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.head(10)


# In[39]:


correlations.tail(10)


# In[40]:


#Duplicate values
features = train_df.columns.values[2:202]
unique_max_train = []
unique_max_test = []
for feature in features:
    values = train_df[feature].value_counts()
    unique_max_train.append([feature, values.max(), values.idxmax()])
    values = test_df[feature].value_counts()
    unique_max_test.append([feature, values.max(), values.idxmax()])


# In[41]:


sns.factorplot('target', data=train_df, kind='count')


# In[42]:


#count of both class(number of classes)
train_df['target'].value_counts()


# In[43]:


#WE seperate the dataset whose target class is belong to class 0
data=train_df.loc[train_df['target'] == 0]
#choose starting 24000 rows
data2=data.loc[:24000]
data2


# In[44]:


#WE seperate the dataset whose target class is belong to class 1
data1=train_df.loc[train_df['target'] == 1]
data1


# In[45]:


#Add both Dataframe data1 and data2 in one dataframe
newdata=pd.concat([data1, data2], ignore_index=True)
newdata


# In[46]:


#Shuffle the Dataframe
newdata=newdata.sample(frac=1)
newdata


# In[47]:


sns.factorplot('target', data=newdata, kind='count')


# In[48]:


#Seperate the input features and store in variable x
x=newdata.iloc[:,2:].values
x=pd.DataFrame(x)
x


# In[49]:


#Seprate the target class and store the class in y variable
y=newdata.iloc[:,1].values
y=pd.DataFrame(y)
y


# In[50]:


#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=100,test_size=0.2)


# In[51]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[52]:


#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=80)
PCA_X_train=pca.fit_transform(X_train)
PCA_X_test=pca.fit_transform(X_test)
explain=pca.explained_variance_ratio_.tolist()
explain


# In[53]:


X_train.shape


# In[54]:


PCA_X_train.shape


# In[55]:


################Logistic Regression#################
from sklearn.linear_model import LogisticRegression

model=LogisticRegression(n_jobs=-1)


# In[56]:


model.fit(PCA_X_train,y_train)


# In[57]:


y_pred=model.predict(PCA_X_test)


# In[58]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[59]:


#find precision ,recall,fscore,support
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))


# In[60]:


(3332+2992)/(3332+931+1012+2992)


# In[61]:


####################Random Forest########################
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=1500,random_state=0)
classifier.fit(PCA_X_train,y_train)


# In[62]:


#Predict from test data
y_pred=classifier.predict(PCA_X_test)


# In[63]:


#Applying confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[64]:


(3347+3006)/(3347+916+998+3006)


# In[65]:


from sklearn.metrics import f1_score
f1_score(y_test,y_pred, average='binary')


# In[66]:


#find precision ,recall,fscore,support
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))


# In[67]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[68]:


test_df = pd.read_csv("C:/Users/kanch/OneDrive/Desktop/project/Project_3/test.csv")
sc.fit(test_df.iloc[:,1:])


# In[69]:


test_df_std=pd.DataFrame(sc.transform(test_df.iloc[:,1:]))


# In[70]:


from sklearn.decomposition import PCA
pca.fit(test_df_std)


# In[71]:


test_df_X=pd.DataFrame(pca.transform(test_df_std))


# In[72]:


predictions=model.predict(test_df_X)


# In[73]:


test_df_X['target']=predictions


# In[74]:


test_df_X.head()


# In[ ]:




