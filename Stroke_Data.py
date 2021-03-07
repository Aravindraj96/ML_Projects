import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
missing_values = ['Unknown']
dataset = pd.read_csv("healthcare-dataset-stroke-data.csv", na_values=missing_values)

cols = dataset.columns[:-1]
colours = ['#000099', '#ffff00']
sns.heatmap(dataset[cols].isnull(), cmap = sns.color_palette(colours))
#Check for NaN in all the columns

# =============================================================================
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# =============================================================================

# =============================================================================
# """
# Assumptions smoking has nothing to do with stroke
# BMI has 201 NAN dropping all the values because BMI is an important factor in stroke
# 
# """
# print(dataset[cols].isnull().sum())
# dataset['smoking_status'].fillna(value="never smoked",inplace = True)
# print(dataset[cols].isnull().sum())
# df_drop_bmi = dataset.dropna()
# """
# Removed the column smoking status  and replacing the bmi with mean values
# """
# df_drop_ss = dataset.drop(['smoking_status'], axis=1)
# df_drop_ss['bmi'].fillna(float(df_drop_ss['bmi'].mean()),inplace=True)
# # dd = ohe.fit_transform(df_drop_ss[0:-1])
# =============================================================================
"""
keeping all data 
"""
df_all=dataset
del df_all['work_type']
del df_all['smoking_status']
del df_all['Residence_type']

df_all['bmi'].fillna(float(df_all['bmi'].mean()), inplace=True)
# df_all['smoking_status'].fillna(value="never smoked",inplace = True)
df_all['gender'] = df_all['gender'].astype('category')
df_all['ever_married'] = df_all['ever_married'].astype('category')
# df_all['work_type'] = df_all['work_type'].astype('category')
# df_all['Residence_type'] = df_all['Residence_type'].astype('category')
# df_all['smoking_status'] = df_all['smoking_status'].astype('category')


df_hm = df_all
print(df_all.dtypes)
df_all=pd.get_dummies(df_all, columns=['gender'], prefix=['gender'])
le = LabelEncoder()
# df_all['gender']=le.fit_transform(df_all['gender'])
df_all['ever_married'] = le.fit_transform(df_all['ever_married'])

df_hm['gender'] = le.fit_transform(df_hm['gender'])
# df_hm['Residence_type'] = le.fit_transform(df_hm['Residence_type'])
# df_hm['work_type'] = le.fit_transform(df_hm['work_type'])
# df_hm['smoking_status'] = le.fit_transform(df_hm['smoking_status'])
df_hm['ever_married'] = le.fit_transform(df_hm['ever_married'])



stroke = df_all['stroke']
df_all.drop(labels=['stroke'],inplace=True,axis=1)
df_all.insert(10, 'stroke', stroke)


fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(df_all.corr(),annot=True)
sns.heatmap(df_all.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap = 'coolwarm', ax=ax)
fig, ax = plt.subplots(figsize=(25,25))
sns.heatmap(df_hm.corr(), annot = True, vmin=-1, vmax=1, center=0, cmap='coolwarm', ax=ax)

print(df_all.describe())


fig, axes = plt.subplots (1, 5, figsize = (15,6), sharey = True)

sns.boxplot(ax=axes[0], data=df_all, x='stroke', y='age', palette='Set2')
sns.boxplot(ax=axes[1], data=df_all, x='stroke', y='heart_disease', palette='Set2')
sns.boxplot(ax=axes[2], data=df_all, x='stroke', y='avg_glucose_level', palette='Set2')
sns.boxplot(ax=axes[3], data=df_all, x='stroke', y='hypertension', palette='Set2')
sns.boxplot(ax=axes[4], data=df_all, x='stroke', y='bmi', palette='Set2')


fig, axes = plt.subplots (1, 5, figsize = (20,6), sharey = True)
df_all.drop(df_all[df_all['age']< 15].index, axis=0, inplace = True)
#df_all.drop(df_all[df_all['avg_glucose_level']< 260].index, axis=0, inplace = True)
sns.boxplot(ax=axes[0], data=df_all, x='stroke', y='age', palette='Set2')
sns.boxplot(ax=axes[1], data=df_all, x='stroke', y='heart_disease', palette='Set2')
sns.boxplot(ax=axes[2], data=df_all, x='stroke', y='avg_glucose_level', palette='Set2')
sns.boxplot(ax=axes[3], data=df_all, x='stroke', y='hypertension', palette='Set2')
sns.boxplot(ax=axes[4], data=df_all, x='stroke', y='bmi', palette='Set2')


x = df_all.iloc[:, 1:-1].values
y = df_all.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(random_state=0)
lrc.fit(x_train, y_train)
y_pred_lrc = lrc.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score, f1_score

cm = confusion_matrix(y_test, y_pred_lrc)

print(cm)

print(accuracy_score(y_test, y_pred_lrc))


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200, random_state=100)

rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)

cm = confusion_matrix(y_test, y_pred_rfc)

print(cm)

print(accuracy_score(y_test, y_pred_rfc))



from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=100, criterion='entropy')
dtc.fit(x_train,y_train)
y_pred_dtc = dtc.predict(x_test)

cm = confusion_matrix(y_test, y_pred_dtc)

print(cm)

print(accuracy_score(y_test, y_pred_dtc))


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=1000)
classifier.fit(x_train, y_train)

y_pred_svc = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred_svc)
print(cm)
print(accuracy_score(y_test, y_pred_svc))



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2 )
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)

cm = confusion_matrix(y_test, y_pred_knn)
print(cm)
print(accuracy_score(y_test, y_pred_knn))


### SVC WiNS THE GAME