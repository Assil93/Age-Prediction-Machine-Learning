# encoding=utf8  
from importlib import reload #doesnt work with python 2.7
import sys 
reload(sys)  
# dont remove this line, is important to handle german umlauts.

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

def knn(cv, X, y):
    # creating odd list of K for KNN
    myList = list(range(1,50))

    # subsetting just the odd ones
    neighbs = filter(lambda x: x % 2 != 0, myList)

    # empty list that will hold cv scores
    knn_acc_train = []
    knn_acc_test = []
    #neighbors_k = range(1,15,1)

    for k in neighbs:
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        count = 0
        accuracy_common_train = 0
        accuracy_common_test = 0

        for train, test in logo.split(X, y, groups=groups):
            #print("%s %s" % (train, test))
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            #print(X_train, X_test, y_train, y_test)
            #print(X_test)
            knn.fit(X_train,y_train)
            accuracy_train = knn.score(X_train, y_train)
            accuracy_common_train +=accuracy_train
            accuracy_test = knn.score(X_test, y_test)
            accuracy_common_test +=accuracy_test
            count +=1
            #print('Accuracy of K-NN classifier on training set: {:.2f}'
            #.format(knn.score(X_train, y_train)))
            #print('Accuracy of K-NN classifyier on test set: {:.2f}'
            #.format(knn.score(X_test, y_test)))
        #print(count)
        mean_accuracy_kNN_train = accuracy_common_train/count
        #print("mean accuracy kNN train" + str(mean_accuracy_kNN_train))
        mean_accuracy_kNN_test = accuracy_common_test/count
        print("mean accuracy kNN test" + str(k) + " " + str(mean_accuracy_kNN_test))
        knn_acc_test.append(mean_accuracy_kNN_test)
        knn_acc_train.append(mean_accuracy_kNN_train)

    #plt.scatter(neighbors_k,knn_acc_train,color='red')
    plt.scatter(neighbs,knn_acc_test,color='blue')
    plt.show()

measures=pd.read_csv('./Daten/measures_0_feat_deleted.csv',  sep=';', decimal=',', error_bad_lines=False)
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
test_ids=  [3, 26, 27, 32, 37, 44, 47, 51, 54, 57, 60, 61, 63, 66, 70, 90, 95, 109, 111, 116, 120, 123, 134, 149, 151]
test_set = (measures.loc[(measures["P-KennungAnonym"].isin(test_ids))])
#test_set.drop(['P-KennungAnonym'], 1, inplace=True)

#count ids used for training set
training_ids = []
import itertools
for id in test_ids:
        if test_ids.index(id) == 0:
            x = np.arange(0,id,1)
        else:
            x = np.arange(test_ids[test_ids.index(id)-1]+1, id, 1)
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

Xtest_set = test_set.drop(columns, 1)
Xtest_set = Xtest_set.values

y = training_set['P-Altersklasse']
y = y.values
ytest_set = test_set['P-Altersklasse']
ytest_set = ytest_set.values
#print(len(y.index))

groups = training_set['P-KennungAnonym']
#print(len(groups.index))
logo = LeaveOneGroupOut()

knn(logo, X,y)


# Compute Generalization Error
knn = neighbors.KNeighborsClassifier(n_neighbors=39)
knn.fit(X, y)
test_set_accuracy = knn.score(Xtest_set,ytest_set)
print(test_set_accuracy)