
#!/usr/bin/python

import sys
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import pprint

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### features_list contains all the features available:
 
features_list = ['poi','salary', 'deferral_payments',  \
               'total_payments', 'loan_advances', 'bonus', 'email_address',  \
              'restricted_stock_deferred', 'deferred_income', \
               'total_stock_value', 'expenses', 'exercised_stock_options', \
               'other', 'long_term_incentive', 'restricted_stock', 'director_fees', \
               'to_messages', 'from_poi_to_this_person', 'from_messages',  \
               'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset:
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#### Total number of observations(rows)

len(data_dict.keys())

### Total Number of persons of interset (POI) and Non POI

# Number of person of interest and not person of interest

num_poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == True:
        num_poi+=1
num_no_poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == False:
        num_no_poi+=1
print ('Number of Person of Interest' + ' ' + str(num_poi))      
print ('Number of Non Person of Interest' + ' ' + str( num_no_poi))

##List of employees 
for employee in data_dict:
    print employee

# Lists of features in the dataset 
data_dict['COLWELL WESLEY']


## Removing Outliers 
'''The total and 'The Travel Agency in The Park' are deleted as they are not name 
 of employees.'''

data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('TOTAL', 0)

'''Lochart Eugene E does not have any information. All his records are missing values. 
So I will remove him from the list.'''

data_dict.pop('LOCKHART EUGENE E', 0)

## Total Number of Observations after removing the outliers

# Checking if the Total and Travel agency are removed from the dataset.
len(data_dict.keys())

### Total Number of POI and Non POI after removing the outliers

num_poi = 0
for user in data_dict:
    if data_dict[user]['poi'] == True:
        num_poi+=1
num_no_poi = 0
for user in data_dict:
    if data_dict[user]['poi'] == False:
        num_no_poi+=1
print ('Number of Persons of Interest' + ' ' + str(num_poi))      
print ('Number of Non Persons of Interest' + ' ' + str( num_no_poi))


## Calculating the percentage of missing values in each feature
def NaN(feature_name):
    NaN=0
    import math
    for employee in data_dict:
        if math.isnan(float(data_dict[employee][feature_name])):
            NaN += 1
    NaN_proportion = round(100*float(NaN)/float(len(data_dict)),2)
    return NaN_proportion
print "Percentage of missing values(NaN) in each feature: "
print "POI:- " + str(NaN('poi')) + "%"
print "Salary:- " + str(NaN('salary')) + "%"
print "Deferral payments:- " + str(NaN('deferral_payments')) + "%"
print "Total payments:- " + str(NaN('total_payments')) + "%"
print "Loan advanced:- " + str(NaN('loan_advances')) + "%"
print "Bonus:- " + str(NaN('bonus')) + "%"
print "Restricted stock deferred:- " + str(NaN('restricted_stock_deferred')) + "%"
print "Deferred Income:- " + str(NaN('deferred_income')) + "%"
print "Total stock value:- " + str(NaN('total_stock_value')) + "%"
print "Expenses:- " + str(NaN('expenses')) + "%"
print "Exercised stock options:- " + str(NaN('exercised_stock_options')) + "%"
print "Other:- " + str(NaN('other')) + "%"
print "Long term incentive:- " + str(NaN('long_term_incentive'))+ "%"
print "Restricted stock:- " + str(NaN('restricted_stock')) + "%"
print "Director fees:- " + str(NaN('director_fees')) + "%"
print "To messages:- " + str(NaN('to_messages')) + "%"
print "From poi to this person:- " + str(NaN('from_poi_to_this_person'))+ "%"
print "From messages:- " + str(NaN('from_messages')) + "%"
print "From this person to poi:- " + str(NaN('from_this_person_to_poi')) + "%"
print "Shared receipt with poi:- " + str(NaN('shared_receipt_with_poi'))+ "%"


# ## Creating New Features

for email in data_dict:          
    fraction_from_poi = float(data_dict[email]['from_poi_to_this_person']) / float(data_dict[email]['to_messages']) 
    if np.isnan(fraction_from_poi):     
        data_dict[email]['fraction_from_poi'] =  0
    else:
        data_dict[email]['fraction_from_poi'] =  round(fraction_from_poi,2)     
    fraction_to_poi = float(data_dict[email]['from_this_person_to_poi']) / float(data_dict[email]['from_messages'])
    if np.isnan(fraction_from_poi):     
        data_dict[email]['fraction_to_poi'] =  0
    else:
        data_dict[email]['fraction_to_poi'] =  round(fraction_to_poi,2)
        
##Check if the two features are included in the dataset        
print data_dict['SUNDE MARTIN']


# ## Selecting Features
'''I will use K-Best to select the best features. To choose the best number of features, 
I am going to check the accuracy, precision and recall for different k values (10, 6, 4, and 2) 
and I will choose the best one.'''

from sklearn.feature_selection import SelectKBest, f_classif
# Since email address is not essential for the prediction, I am not listing it with the features.
features_list = ['poi','salary', 'deferral_payments','total_payments', 'loan_advances', 
                 'bonus','restricted_stock_deferred','deferred_income','total_stock_value',\
                 'expenses', 'exercised_stock_options','other', 'long_term_incentive',\
                 'restricted_stock','director_fees','to_messages', 'from_poi_to_this_person',\
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi',\
                 'fraction_from_poi', 'fraction_to_poi']

data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

def best_features_cal(data_dict, features_list, k):
    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} Best Features: {1}\n".format(k, k_best_features)
    return k_best_features

##K= 10
best_features = best_features_cal(data_dict, features_list, 10)
selected_features_list_1 = [labels] + best_features.keys()

##K=6
best_features = best_features_cal(data_dict, features_list, 6)
selected_features_list_2 = [labels] + best_features.keys()

##k= 4
best_features = best_features_cal(data_dict, features_list, 4)
selected_features_list_3 = [labels] + best_features.keys()

##k= 2
best_features = best_features_cal(data_dict, features_list, 2)
selected_features_list_4 = [labels] + best_features.keys()

### 1. For k= 10

##Checking Accuracy with the best 10 Features selected
selected_features_list_1 = ['poi', 'salary','total_payments', 'bonus', 'deferred_income',\
                           'total_stock_value', 'exercised_stock_options', 'long_term_incentive',\
                           'restricted_stock', 'shared_receipt_with_poi','fraction_to_poi']
from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier()
test_classifier(clf1, data_dict, selected_features_list_1)

### 2. For K=6

##Checking Accuracy with the best 6 Features selected
selected_features_list_2 = ['poi', 'salary', 'bonus', 'deferred_income','total_stock_value',\
                              'exercised_stock_options','fraction_to_poi']
from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier()
from tester import test_classifier
test_classifier(clf1, data_dict, selected_features_list_2)

### 3. For K = 4

##Checking Accuracy with the best 4 Features selected
selected_features_list_3 = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options']
from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier()
from tester import test_classifier
test_classifier(clf1, data_dict, selected_features_list_3)


### 4. For K = 2


##Checking Accuracy with the best 2 Features selected
selected_features_list_4 = ['poi', 'total_stock_value', 'exercised_stock_options']
from sklearn.tree import DecisionTreeClassifier
clf3 = DecisionTreeClassifier()
from tester import test_classifier
test_classifier(clf3, data_dict, selected_features_list_4)


### The best accuracy is attended with K=10.

### Effect of the New variables

'''The selection was done including the new features create. I want to check the effect of 
the new variables by removing it from the list. '''

##selecting the best 10 features by removing the two new features created(removing fraction_from_poi and fraction_to_poi)

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',\
                 'restricted_stock_deferred', 'deferred_income','total_stock_value', 'expenses', \
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',\
                 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',\
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
                
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
selector=SelectKBest(f_classif,k=10)
selector.fit(features,labels)
print "10 best features: "
for i in list(selector.get_support(indices=True)):
    print features_list[i + 1], ' - Feature Index: ', i + 1
print " "

## Cheecking difference removing the new created features
selected_features_list_5 = ['poi', 'salary','total_payments','loan_advances', 'bonus', \
                            'deferred_income','total_stock_value','exercised_stock_options',\
                            'long_term_incentive','restricted_stock', 'shared_receipt_with_poi']
from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier()
from tester import test_classifier
test_classifier(clf1, data_dict, selected_features_list_5)

# Accuracy, Precision and Recall is slighlty higher when new variables are added.

### Store to my_dataset for easy export below.
my_dataset = data_dict

# ## Algorithms 

### 1. Decision Tree Classifier

'''From the above tests, using Decision tree classifier, the best accuracy, precision
and recall was found when k is 10. Hence, I will use the best 10 features.'''

#selected_features_list = ['poi', 'salary','total_payments', 'bonus', 'deferred_income',\
 #                'total_stock_value', 'exercised_stock_options', 'long_term_incentive',\
  #               'restricted_stock', 'shared_receipt_with_poi','fraction_to_poi']


selected_features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options']
data = featureFormat(data_dict, selected_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
test_classifier(dtc, my_dataset, selected_features_list)

### 2. Random Forest
data = featureFormat(data_dict, selected_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.ensemble import RandomForestClassifier
from tester import test_classifier
rf = RandomForestClassifier()
test_classifier(rf, my_dataset, selected_features_list)


### 3. KNeighbors
data = featureFormat(data_dict, selected_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
test_classifier(knn, my_dataset, selected_features_list)


####trying scaling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
knn = KNeighborsClassifier()
clf = Pipeline(steps=[('scale', MinMaxScaler()),('estimate', KNeighborsClassifier())])
test_classifier(clf, my_dataset, features_list)
 

### 4. AdaBoostClassifier
data = featureFormat(data_dict, selected_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.ensemble import AdaBoostClassifier
from tester import test_classifier
adc = AdaBoostClassifier()
test_classifier(adc, my_dataset, selected_features_list)


# The best accuracy and  precision is attended with kNeighbors.

### Tuning to Get the Best Parameter
##Tuning Random Forest

from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(labels, test_size=0.3, random_state=42)
                                     
para= {'max_depth': [4,6],
              'min_samples_split':[2,3,4,6],
              'n_estimators':[10,20,30],
              'min_samples_leaf':[1,2,3,4]}
rf = RandomForestClassifier()
rft = GridSearchCV(rf, para, scoring = 'f1', cv=sss)
rft.fit(features, labels)
rftc = rft.best_estimator_
test_classifier(rftc, my_dataset, selected_features_list)

## Tuning AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
sss = StratifiedShuffleSplit(labels,10, test_size=0.3, random_state=42)
                                    
depth = []
for i in range(5):
    depth.append(DecisionTreeClassifier(max_depth=(i+1)))
para = {'base_estimator': depth, 
             'n_estimators': [50,100]}

adclf= AdaBoostClassifier()                            
ad = GridSearchCV(adclf, para, scoring='f1', cv=sss)
ad.fit(features, labels)
adcf = ad.best_estimator_
test_classifier(adcf, my_dataset, selected_features_list)

##Tuning KNeighbors 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
sss = StratifiedShuffleSplit(labels, 10, test_size=0.3, random_state=42)
estimators = [('scale', MinMaxScaler()), ('knn', knn)]
pipeline = Pipeline(estimators)
para = {'knn__n_neighbors': range(1,20),'knn__algorithm':('ball_tree','kd_tree','brute','auto')}
knngrd = GridSearchCV(pipeline, para, scoring = 'f1', cv=sss)
knngrd.fit(features, labels)
knn = knngrd.best_estimator_
test_classifier(knn, my_dataset, selected_features_list)

clf = ad.best_estimator_
test_classifier(clf, my_dataset, selected_features_list)
dump_classifier_and_data(clf, my_dataset,selected_features_list)

