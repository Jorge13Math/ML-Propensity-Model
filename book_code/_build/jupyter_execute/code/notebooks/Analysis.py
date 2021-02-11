# This notebook help to create a model to find next potential clients 

# Imports

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath('../'))
sys.path.append(os.path.realpath('../../'))
from scripts.create_dataset import genererate_dataframe
from scripts.utils import Utils
from scripts.preprocess_data import Preprocces


# Loading data and overview

path = '../../data/'

df_data,df_client = genererate_dataframe(path)

df_data.head()

df_data.columns

df_data.dtypes

df = df_data.copy()

df

df.shape

df_client.head()

df_client.shape

df_client.isna().sum()

Creating a new column called client that contain:

**1:** is a client 
**0:** is not a client  

Id = df_client.id.values.tolist()

df['client'] = df.id.apply(lambda x: 1 if x in Id else 0)

Now the datatset is ready to be analyzed

df.head()

# EDA

In this section the data will be analyzed. This process is very important as it helps to find important patterns and characteristics in the data.

df.describe().T

df.info()

df.shape

Creating a list type variable called **col_remove**, in which the features that are not important for our goal will be added

col_remove = ['id']

## Analysis of Missing Values 

df.isna().sum()

Loading a class called Utils, this class helps to vizualize the data

utils = Utils()

utils.plot_variables_nan(df)

utils.df_nan

Removing the **riesgo** variable since it has more than 99% the NaN

df = df.drop(columns=['riesgo'])

df.shape

## Analysis of target value

df[['client']].hist()
plt.ylabel('Count')
plt.show()

df.client.value_counts()

print(f"Percentage of target 0 : {round(df.client.value_counts().values[0] * 100 / len(df))} %" )

print(f"Percentage of target 1 : {round(df.client.value_counts().values[1] * 100 / len(df))} %")

## Analysis of Categorical Variables

df.select_dtypes(include ='object').columns

utils.plot_variable(df,'tipo_empresa')
utils.plot_variable_per_target(df,'tipo_empresa','client')

This dataframe represents the percentage of the customer per variable.

**Columns:** 1: is a client, 0: is not a client

**Row:** Variable value

This percentage will be taken into account for the selection of variables to eliminate.

This process is carried out with all variables.

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

This variable **tipo_empresa** is added to col_remove since there is no difference between customer and non-customer. 99% of clients are companies

col_remove.append('tipo_empresa')

utils.plot_variable(df,'comunidadautonoma')
utils.plot_variable_per_target(df,'comunidadautonoma','client')

df_community = utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)
df_community.fillna(value=0,inplace=True)
df_community

df_community[df_community['1']>3]['1'].sum()

Creating a new variable called **community_not_client**. This variable contains the communities that do not belong to 85% of the clients

community_not_client = list(df_community[df_community['1']<3].index)
community_not_client

utils.plot_variable(df,'capitalpueblo')
utils.plot_variable_per_target(df,'capitalpueblo','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

df_actcnae = utils.plot_variable(df,'actcnae',True)

df_actcnae

df_actcnae_class= utils.plot_variable_per_target(df,'actcnae','client',True)

df_actcnae_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

Adding the **actcnae** variable since the **cnae_mercado** variable will be taken into account since it encompasses a **actcnae**

col_remove.append('actcnae')

utils.plot_variable(df,'mercado')
utils.plot_variable_per_target(df,'mercado','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

Adding the **mercado** variable since the **cnae_mercado** variable will be taken into account since it encompasses a **mercado**

col_remove.append('mercado')

utils.plot_variable(df,'cnae_mercado')
utils.plot_variable_per_target(df,'cnae_mercado','client')


df_cnae_market =utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)
df_cnae_market.fillna(value=0, inplace=True)
df_cnae_market

df_cnae_market[df_cnae_market['1']>3]['1'].sum()

Creating a new variable called **cnae_market_not_client**. This variable contains the cnae_market that do not belong to 85% of the clients

cnae_market_not_client = df_cnae_market[df_cnae_market['1']<3].index
cnae_market_not_client

utils.plot_variable(df,'tipooficina')
utils.plot_variable_per_target(df,'tipooficina','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

df_fundacion = utils.plot_variable(df,'anyofundacion',True)


df_fundacion['time'] = pd.to_datetime(df_fundacion.anyofundacion, format='%d/%m/%Y')

df_fundacion

df_fundacion.groupby(df_fundacion['time'].dt.year,as_index=False)['Count'].agg(['sum']).plot(rot=90,grid=True,figsize=(10, 5),legend=True)
plt.title(f"Number of company for year")
plt.ylabel('Count')
plt.show()

df_fundacion_class = utils.plot_variable_per_target(df,'anyofundacion','client',True)

df_fundacion_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

df_fundacion_class['time'] = pd.to_datetime(df_fundacion_class.index, format='%d/%m/%Y')

df_fundacion_class.groupby(df_fundacion_class['time'].dt.year,as_index=False)['0','1'].agg(['sum']).plot(rot=90,grid=True, figsize=(10, 5),legend=True)
plt.title(f"Number of company for year and class")
plt.ylabel('Count')
plt.show()

Adding the **anyofundacion** variable since further it will be apply a transformation to this variable

col_remove.append('anyofundacion')
col_remove

utils.plot_variable(df,'tendenciaempleados')
utils.plot_variable_per_target(df,'tendenciaempleados','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

The **tendenciaempleados** variable is eliminated since it has more than 53% of missing values and it is dificult to impute all values.

col_remove.append('tendenciaempleados')

utils.plot_variable(df,'tendenciaingresos')
utils.plot_variable_per_target(df,'tendenciaingresos','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

The **tendenciaingresos** variable is eliminated since it has more than 63% of missing values and it is dificult to impute all values.

col_remove.append('tendenciaingresos')

utils.plot_variable(df,'universo')
utils.plot_variable_per_target(df,'universo','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

## Analysis of Numerical Variables

df_numeric = df.select_dtypes(include =['float64','int64'])

df_numeric

df_numeric.columns[1:]

df_numeric[df_numeric.columns[1:-1]].hist(grid=True, figsize=(20, 20))
plt.show()

utils.plot_variable(df_numeric,'enpoligono')
utils.plot_variable_per_target(df_numeric,'enpoligono','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

utils.plot_variable(df_numeric,'tipo_de_zona')
utils.plot_variable_per_target(df_numeric,'tipo_de_zona','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

utils.plot_variable(df_numeric,'nse')
utils.plot_variable_per_target(df_numeric,'nse','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

utils.plot_variable(df_numeric,'flotaturismos')
utils.plot_variable_per_target(df_numeric,'flotaturismos','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

utils.plot_variable(df_numeric,'flotafurgonetas')
utils.plot_variable_per_target(df_numeric,'flotafurgonetas','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

utils.plot_variable(df_numeric,'flotacamiones')
utils.plot_variable_per_target(df_numeric,'flotacamiones','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

The **flotacamiones** is added to col_remove since there is no difference between customer and non-customer. 97% of clients are not truck fleet

col_remove.append('flotacamiones')

utils.plot_variable(df_numeric,'flotaautobuses')
utils.plot_variable_per_target(df_numeric,'flotaautobuses','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

The **flotaautobuses** is added to col_remove since there is no difference between customer and non-customer. 99% of clients are not bus fleet

col_remove.append('flotaautobuses')

utils.plot_variable(df_numeric,'existe_import')
utils.plot_variable_per_target(df_numeric,'existe_import','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

The **existe_import** is added to col_remove since there is no difference between customer and non-customer. 94% of clients do not import

col_remove.append('existe_import')

utils.plot_variable(df_numeric,'existe_export')
utils.plot_variable_per_target(df_numeric,'existe_export','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

utils.plot_variable(df_numeric,'existe_importexport')
utils.plot_variable_per_target(df_numeric,'existe_importexport','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

The **existe_importexport** is added to col_remove since there is no difference between customer and non-customer. 95% of clients do not import or export

col_remove.append('existe_importexport')

utils.plot_variable(df_numeric,'existe_email')
utils.plot_variable_per_target(df_numeric,'existe_email','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

utils.plot_variable(df_numeric,'es_ecommerce')
utils.plot_variable_per_target(df_numeric,'es_ecommerce','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

The **es_ecommerce** is added to col_remove since there is no difference between customer and non-customer. 99% of clients are not ecommerce

col_remove.append('es_ecommerce')

utils.plot_variable(df_numeric,'rangoventas')
utils.plot_variable_per_target(df_numeric,'rangoventas','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

utils.plot_variable(df_numeric,'numoficinas')
utils.plot_variable_per_target(df_numeric,'numoficinas','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

The **numoficinas** variable is eliminated since it more than 91% of clients does not office

col_remove.append('numoficinas')

utils.plot_variable(df_numeric,'impactos_publicidad')
utils.plot_variable_per_target(df_numeric,'impactos_publicidad','client')

utils.df_class.apply(lambda x: (x / x.sum()*100)).sort_values(by=['1'],ascending=False)

sns.boxplot(x='client',y='rangoempleados',data=df)
plt.title('rangoempleados without logarithmic smoothing')
plt.show()

sns.boxplot(x='client',y='habitantes_municipio',data=df)
plt.title('habitantes_municipio without logarithmic smoothing')
plt.show()

sns.boxplot(x='client',y='impactos_publicidad',data=df)
plt.title('impactos_publicidad')
plt.show()

df_log = df[['habitantes_municipio']].applymap(lambda x:np.log(x) if x>0 else x)
df_log['client'] = df['client']

sns.boxplot(x='client',y='habitantes_municipio',data=df_log)
plt.title('habitantes_municipio with logarithmic smoothing')
plt.show()


The **habitantes_municipio** variable is eliminated since there is no difference in population between clients and non-clients   

col_remove.append('habitantes_municipio')

col_remove

## Correlation between variables

df_num = df.select_dtypes(include = ['float64','int64']).copy()

corr =  df_num.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


f, ax = plt.subplots(figsize=(11, 9))


cmap = sns.diverging_palette(240, 10, n=9)


sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Correlation between variables')
plt.show()

The following variables will be added to the list of variables to be eliminated since only the **comunidadautonoma** will be taken into account

for var in ['codpostal','municipio','provincia']:
    col_remove.append(var)

col_remove

# Preprocess Data 

In this session, data is processed, cleaned, and training and test sets are created to train a propensity model to find potential customers.

df.columns

preprocess = Preprocces(df)

Cleaning dataframe

df_clean = preprocess.clean_dataframe(col_remove)

df_clean.head()

df_clean.describe().T

df_clean[df_clean['client']==0].groupby(['comunidadautonoma'])['client'].agg(['count']).sort_values(by ='count',ascending=False).loc[community_not_client]

df_clean[df_clean['client']==0].groupby(['cnae_mercado'])['client'].agg(['count']).sort_values(by ='count',ascending=False).loc[cnae_market_not_client]

utils.plot_variables_nan(df_clean)

utils.df_nan

## 1- Imputation of Missing Values in numerical Data 

df_clean = preprocess.fill_numerical_na(['rangoempleados','rangoventas','year'],df_clean)

df_clean.isna().sum()

df_clean.describe().T

## 2- Encoded Categorical variables to numeric

df_clean.select_dtypes('object').columns

df_encode = pd.get_dummies(df_clean, columns = df_clean.select_dtypes('object').columns,prefix = 'is',drop_first=True)

df_encode.head()

df_encode.columns

df_encode.shape

## 3- Imputation of Missing Values in categorical Data with KnnImput

df_encode_clean = preprocess.fill_categorical_na(df_encode)

df_encode_clean.isna().sum()

df_encode_clean

Cheking that there is not NaN

df_encode_clean.isna().sum()

## 4 - Selecting the companies that are not potential clients

Two filters are applied to the dataframe to select the non-potential customers based on the following lists **cnae_market_not_client** and **community_not_client**. 

A dataframe called **df_not_client** was created 

df_clean[(df_clean['cnae_mercado'].isin(cnae_market_not_client)) & (df_clean['comunidadautonoma'].isin(community_not_client))][df_clean['client']==0]

df_not_client = df_clean[(df_clean['cnae_mercado'].isin(cnae_market_not_client)) & (df_clean['comunidadautonoma'].isin(community_not_client))][df_clean['client']==0]

df_not_client

index_not_client = list(df_not_client.index)

df_clean[(df_clean['cnae_mercado'].isin(cnae_market_not_client)) & (df_clean['comunidadautonoma'].isin(community_not_client))][df_clean['client']==1].shape

There are only 107 customers that meet the filter mentioned above, this represents 2% of customers

## 5- Creating dataset to train a model 

df_encode_clean.head()

df_encode_clean[df_encode_clean['client']==1]

df_encode_clean.loc[index_not_client]

df_train = pd.concat([df_encode_clean[df_encode_clean['client']==1],df_encode_clean.loc[index_not_client]])

df_test = df_encode_clean.loc[~df_encode_clean.index.isin(df_train.index)]

df_test = df_test.drop(['client'],axis=1)

df_train.client.value_counts()

Now we have a training set, in which we have clients and non-clients

## 6- Take a look to the data

from sklearn import preprocessing
from sklearn.manifold import TSNE

X = df_train.drop(['client'],axis=1)
Y = df_train.client

X.shape,Y.shape

scaler = preprocessing.StandardScaler().fit(X)

Standardizing features by removing the mean and scaling to unit variance

Xs = scaler.transform(X)

Xs

T-SNE is a tool to visualize high-dimensional data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.
by: **Scikit-learn**

X_tsne = TSNE(n_components=2, random_state=0,perplexity=50).fit_transform(Xs)

df_tsne = pd.DataFrame({'comp_1':X_tsne[:, 0],'comp_2':X_tsne[:, 1]})
df_tsne['client'] = Y.values.tolist()

df_tsne

plt.figure(figsize=(10, 10))
sns.scatterplot(x="comp_1", y="comp_2", data=df_tsne, hue="client")
plt.title('Customers and non-customers in two dimensions')
plt.show()

In the previous graph it can be seen that customers and non-customers are separated and grouped together, you can see certain clusters that contain customers and non-customers but this is because there are 2% of customers that have the same characteristics as non-potential customers.

This chart also helps verify that customer segmentation was done correctly.

## 7 Split data to train the model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2,shuffle=True,random_state =0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Train a Model

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

### Try Difference Clasifier 

clfs = {
    'mnb': MultinomialNB(),
    'gnb': GaussianNB(),
    'mlp1': MLPClassifier(),
    'mlp2': MLPClassifier(activation='logistic', solver='adam', alpha=0.1, hidden_layer_sizes=(100, ), learning_rate= 'adaptive' ,random_state=1, max_iter=300),
    'ada': AdaBoostClassifier(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'gbc': GradientBoostingClassifier(),
    'lr': LogisticRegression(),
    'xgb': XGBClassifier()
}

f1_scores = dict()
for clf_name in clfs:
    print(f'Clasifier : {clf_name}')
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1_scores[clf_name] = f1_score(y_pred, y_test)

f1_scores

Apparently all the models work correctly, what may be happening is that it is falling into an overfitting

best_model = max(f1_scores, key=f1_scores.get)
print(f'The best model is {best_model} with f1_score :{f1_scores[best_model]}')

### Try MLP 

It will be tested with MLP to show that it is falling into overfitting

clf =MLPClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)

proba

print(classification_report(pred, y_test))

metrics.plot_confusion_matrix(clf, X_test, y_test,cmap=plt.cm.Blues)

The prediction is made in the test set

result_mlp = np.argmax(clf.predict_proba(df_test.to_numpy()), axis=1)

result_mlp[result_mlp==0]

result_mlp[result_mlp==1].shape[0]*100/result_mlp.shape[0]

This result says that 99% are potential customers, which does not make sense

### Try LogisticRegression.

It will be tested with Logistic Regression to show a good result and it uses a hyperparameter C to avoid overfitting adding stronger regularization.

lr = LogisticRegression(C=0.001).fit(X_train, y_train)

lr.predict_proba(X_test)

print(classification_report(lr.predict(X_test), y_test))

accuracy_score(lr.predict(X_test), y_test)

The table above shows what is the performance of the model by class. The model achieve 87% the accuracy for the class 1 and 85% for the class 0, in general the has 86% of accuracy.

metrics.plot_confusion_matrix(lr, X_test, y_test,cmap=plt.cm.Blues)

result_lr = np.argmax(lr.predict_proba(df_test.to_numpy()), axis=1)

result_lr[result_lr==0].shape[0]*100/result_lr.shape[0]

result_lr[result_lr==1].shape[0]*100/result_lr.shape[0]

proba_pred = lr.predict_proba(X_test)
fpr, tpr, thresh = roc_curve(y_test,proba_pred[:,1])
auc = roc_auc_score(y_test,proba_pred[:,1])

plt.figure()
lw=2
plt.plot(fpr,tpr, color='darkorange',lw=lw,label='ROC curve (area=%0.2f)' %auc)
plt.plot([0,1],[0,1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()

The auc of roc curve achieve 94% it is a very good result for the model 

# Choosing the best model

With the results obtained, the best model is lr for the train set

The model was called lr.


import pickle

Saving a model into a pickle, this avoid to train the model again.

try:
    clf_final = pickle.load(open('../../models/model_lr.pickle','rb'))
except:
    pickle.dump(lr, open('../../models/model_lr.pickle', 'wb'))

# Making Prediction 

result = clf_final.predict_proba(df_test.to_numpy())

result

Creating a new dataframe called df_result to see and select the potencial clients

df_result = df_clean.loc[df_test.index].drop('client',axis=1).copy()

Creating new columns with probabilty for each class

df_result['proba_0'] = result[:,0].tolist()
df_result['proba_1'] = result[:,1].tolist()

Sort dataframe with the highest probability to class 1 that represent the potencial client

df_result.sort_values(by='proba_1',ascending=False).head(10)

list(set(df_clean.comunidadautonoma.unique().tolist())-set(community_not_client))

list(set(df_clean.cnae_mercado.unique().tolist())-set(cnae_market_not_client))

## Criteria final in the choice of companies

df_result[(~df_result.impactos_publicidad.isin(community_not_client)) & (df_result.impactos_publicidad.isin([0,1,2,3])) & (df_result.proba_1>0.95)].sort_values(by='proba_1',ascending=False)

df_result[(~df_result.impactos_publicidad.isin(community_not_client)) & (df_result.impactos_publicidad.isin([0,1,2,3])) & (df_result.proba_1>0.95)].sort_values(by='proba_1',ascending=False).to_csv('../../result/predict_potencial_client.csv')


```{toctree}
:hidden:
:titlesonly:


../../Explanation
```
