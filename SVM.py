# encoding=utf8  
from importlib import reload #doesnt work with python 2.7
import sys 
reload(sys)  
#sys.setdefaultencoding('utf8') # dont remove this line, is important to handle german umlauts.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, neighbors, cross_validation, tree, model_selection
import matplotlib.pyplot as plt
#tree_visualization
from IPython.display import Image 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pydotplus
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler

measures=pd.read_csv('./Daten/measures.csv',  sep=';', decimal=',', error_bad_lines=False)
predict=pd.read_csv('./Daten/to_predict.csv', sep=';',  decimal=',', error_bad_lines=False)



#fill missing data with mean value, dont drop the lines with missing data! 
measures_head = list(measures.head(0))
measures_head.remove('P-KennungAnonym')
measures_head.remove('P-Altersklasse')
measures_head.remove('P-Geschlecht')
for x in measures_head:
    measures[x].fillna(measures[x].mean(), inplace=True)
    predict[x].fillna(predict[x].mean(), inplace=True)





#knnOutput = KNN(k=5).complete(measures)
#knnOutput = KNN(k=5).complete(predict)


#*********************Split Data Frame in Test and Training Sets*****************************************

#Ihre Testmenge sind die Personen [3, 26, 27, 32, 37, 44, 47, 51, 54, 57, 60, 61, 63, 66, 70, 90, 95, 109, 111, 116, 120, 123, 134, 149, 151] 
predic_ids=  [3, 26, 27, 32, 37, 44, 47, 51, 54, 57, 60, 61, 63, 66, 70, 90, 95, 109, 111, 116, 120, 123, 134, 149, 151]
predict_set = (measures.loc[(measures["P-KennungAnonym"].isin(predic_ids))])
#predict_set.drop(['P-KennungAnonym'], 1, inplace=True)

#count ids used for training set
training_ids = []
import itertools
for id in predic_ids:
        if predic_ids.index(id) == 0:
            x = np.arange(0,id,1)
        else:
            x = np.arange(predic_ids[predic_ids.index(id)-1]+1, id, 1)
        training_ids.append(x)
training_ids.append([152, 153])
training_ids = list(itertools.chain.from_iterable(training_ids))
#print(training_ids)

#create training set
training_set = (measures.loc[(measures["P-KennungAnonym"].isin(training_ids))])
print(len(training_set.index))
#print(training_set)
from sklearn.model_selection import LeaveOneGroupOut
columns = ['P-KennungAnonym', 'P-Altersklasse']
X = training_set.drop(columns, 1)
X = X.values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

y = training_set['P-Altersklasse']
y = y.values
#print(len(y.index))

groups = training_set['P-KennungAnonym']
#print(len(groups.index))
logo = LeaveOneGroupOut()

# empty list that will hold cv scores
svc_acc_train = []
svc_acc_test = []

for i in list([ 'sigmoid', 'poly' , 'linear', 'rbf']):
    if (i=='poly'):
        clf = SVC(kernel=i, C=1,degree=1, random_state=0)
    elif (i == 'rbf'):
        clf = SVC(kernel=i, C=1,gamma = 0.9, random_state=0)
    else:
        clf = SVC(kernel=i, C=1, random_state=0)
    count = 0
    accuracy_common_train = 0
    accuracy_common_test = 0

    for train, test in logo.split(X, y, groups=groups):
        #print("%s %s" % (train, test))
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        #print(X_train, X_test, y_train, y_test)
        #print(X_test)
        clf.fit(X_train,y_train)
        accuracy_train = clf.score(X_train, y_train)
        accuracy_common_train +=accuracy_train
        accuracy_test = clf.score(X_test, y_test)
        accuracy_common_test +=accuracy_test
        count +=1
        #print('Accuracy of K-NN classifier on training set: {:.2f}'
        #.format(knn.score(X_train, y_train)))
        #print('Accuracy of K-NN classifyier on test set: {:.2f}'
        #.format(knn.score(X_test, y_test)))
    #print(count)
    mean_accuracy_clf_train = accuracy_common_train/count
    print("mean accuracy SVC train " + i + " "+ str(mean_accuracy_clf_train))
    mean_accuracy_clf_test = accuracy_common_test/count
    print("mean accuracy SVC test " + i + " " +  str(mean_accuracy_clf_test))
    #knn_acc_test.append(mean_accuracy_kNN_test)
    #knn_acc_train.append(mean_accuracy_kNN_train)
