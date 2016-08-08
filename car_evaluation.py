# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 12:02:54 2016

@author: Kiyoko
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

'''
Step1: Define the problem and the metric
- Did you specify the type of data analytic question (e.g. exploration, 
  association causality, class) before touching the data?
- Did you define the metric for success before beginning?
- Did you understand the context for the question and the scientific or business application?
- Did you record the experimental design?
- Did you consider whether the question could be answered with the available data?
'''

'''
Step2: Checking the data
Bad data leads to bad models. Always check your data first.
- is there anything wrong with the data?
- are there any quirks with the data?
- do I need to fix or remove any of the data?
'''

### Read data into pandas dataframe and look for missing data
car_data = pd.read_csv('car.csv', na_values=['NA'])
car_data.head()
### check statistics summary with known data range
car_data.describe()
### Putting '5' for '5+' features
car_data.loc[car_data['doors'] == '5more', 'doors'] = '5'
car_data.loc[car_data['persons'] == 'more', 'persons'] = '5'
### print out unique features and frequency per class for verification
for column_index, column in enumerate(car_data.columns):
    print '{}\t{}'.format(column, car_data[column].unique())
    print car_data.loc[car_data['class'] == 'vgood', column].value_counts()

'''
ways to handle categorical data
1. Fill missing values first: df = df.fillna( 'NA' )
2. Separate out categorical vs numerical columns
cat_col = [ 'a', 'list', 'of', 'categorical', 'column', 'names' ]
cat_df = df[ cat_col ]
cat_dict = cat_df.T.to_dict().values()

num_col = [ 'UserID, 'YOB', 'votes', 'Happy']
cat_df = df.drop( num_col, axis = 1 ]
3. pd.get_dummies(df)
'''
num_col = ['doors', 'persons']
y_col = ['class']
cat_car_data = car_data.drop(num_col, axis=1)
cat_car_data = cat_car_data.drop(y_col, axis=1)
cat_car_data = pd.get_dummies(cat_car_data)
num_car_data = car_data[num_col].apply(lambda x: pd.to_numeric(x))
x_data = pd.concat([num_car_data, cat_car_data], axis=1)
print x_data.head()

sb.pairplot(car_data.dropna())

'''
Step5: Model
- split training & test data

'''
### represent input data as a list of lists
features = x_data.values
labels = car_data['class'].values
print features[:5]
print labels[:5]

### split data
from sklearn.cross_validation import train_test_split
(features_train, features_test, labels_train, labels_test) = train_test_split(features, labels, train_size=0.75, random_state=1)

### apply algorithm
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

clf = GaussianNB()
score = cross_val_score(clf, features, labels)
print "GaussianNB: ", score.mean()

clf = SVC()
score = cross_val_score(clf, features, labels)
print "svm: ", score.mean()

clf = AdaBoostClassifier()
score = cross_val_score(clf, features, labels)
print "AdaBoost: ", score.mean()

clf = RandomForestClassifier()
score = cross_val_score(clf, features, labels)
print "Random Forest: ", score.mean()

clf = KNeighborsClassifier()
score = cross_val_score(clf, features, labels)
print "KNN: ", score.mean()

'''
GaussianNB:     0.537
svm:            0.717*
AdaBoost:       0.725*
Random Forest:  0.770*
KNN:            0.752*
'''

### check for overfitting (score dependency on the training/testing subsets)
model_accuracies = []
for repetition in range(100):
    (features_train, features_test, labels_train, labels_test) = train_test_split(features, labels, train_size=0.75)
    clf = RandomForestClassifier()
    clf.fit(features_train, labels_train)
    acc = clf.score(features_test, labels_test)
    model_accuracies.append(acc)
    
sb.distplot(model_accuracies) ### gives good normal distribution with mu ~0.96

### evaluate 10-fold cross-validation --> StratifiedKFols
from sklearn.cross_validation import cross_val_score, StratifiedKFold
clf = RandomForestClassifier()
cv_scores = cross_val_score(clf, features, labels, cv=10)
print len(list(cv_scores))
sb.distplot(cv_scores)

### parameter tuning with grid search
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

select = SelectKBest()
steps = [('feature_selection', select), ('random_forest', clf)]
parameters = dict(feature_selection__k=[10, 15, 'all'], random_forest__n_estimators=[5, 10, 15, 20], random_forest__criterion=['gini', 'entropy'], random_forest__max_features=[1,2,3,4], random_forest__min_samples_split=[2,3,4,5])
pipeline = Pipeline(steps)
cv = StratifiedKFold(labels, n_folds=10)
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=cv)
grid_search.fit(features, labels)
print 'Best score: {}'.format(grid_search.best_score_)
print 'best parameters: {}'.format(grid_search.best_params_)
'''
Best score: 0.861
best parameters: {'random_forest__min_samples_split': 4, 'random_forest__n_estimators': 20, 'feature_selection__k': 'all', 'random_forest__max_features': 4, 'random_forest__criterion': 'entropy'}
'''

'''
Finalize the model
'''
### plot the cross-validation scores
clf = RandomForestClassifier(max_features=4, min_samples_split=4, criterion='entropy', n_estimators=20)
cv_scores = cross_val_score(pipeline, features, labels, cv=10)
sb.boxplot(cv_scores)
sb.stripplot(cv_scores, jitter=True, color='red')

### show some predictions
(features_train, features_test, labels_train, labels_test) = train_test_split(features, labels, train_size=0.75, random_state=1)
(features_train, features_test, labels_train, labels_test) = train_test_split(features, labels, train_size=0.75, random_state=1)
clf.fit(features_train, labels_train)

for input_features, pred, actual in zip(features_test[:10], clf.predict(features_test[:10]), labels_test[:10]):
    print '{}\t --> \t{}\t(Actual: {})'.format(input_features, pred, actual)



