# encoding=utf8  
#from importlib import reload #doesnt work with python 2.7
import sys 
reload(sys)  
sys.setdefaultencoding('utf8') # dont remove this line, is important to handle german umlauts.

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

def desicion_tree(cv, X, y):
    # empty list that will hold cv scores
    destree_acc_train = []
    destree_acc_test = []
    depth = range(1,11,1)

    for d in depth:
        dtree = tree.DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=0)
        count = 0
        accuracy_common_train = 0
        accuracy_common_test = 0

        for train, test in logo.split(X, y, groups=groups):
            #print("%s %s" % (train, test))
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            #print(X_train, X_test, y_train, y_test)
            #print(X_test)
            dtree.fit(X_train,y_train)
            accuracy_train = dtree.score(X_train, y_train)
            accuracy_common_train +=accuracy_train
            accuracy_test = dtree.score(X_test, y_test)
            accuracy_common_test +=accuracy_test
            count +=1
            #print('Accuracy of K-NN classifier on training set: {:.2f}'
            #.format(dtree.score(X_train, y_train)))
            #print('Accuracy of K-NN classifyier on test set: {:.2f}'
            #.format(dtree.score(X_test, y_test)))
        #print(count)
        mean_accuracy_dtree_train = accuracy_common_train/count
        #print("mean accuracy dtree train" + str(mean_accuracy_dtree_train))
        mean_accuracy_dtree_test = accuracy_common_test/count
        print("mean accuracy dtree train" + str(d) + " " + str(mean_accuracy_dtree_train))
        print("mean accuracy dtree test" + str(d) + " " + str(mean_accuracy_dtree_test))
        destree_acc_test.append(mean_accuracy_dtree_test)
        destree_acc_train.append(mean_accuracy_dtree_train)
    plt.scatter(depth,destree_acc_train,color='red')
    plt.scatter(depth,destree_acc_test,color='blue')
    plt.show()

#Read data
measures=pd.read_csv('./Daten/measures_0_ls.csv',  sep=';', decimal=',', error_bad_lines=False)
predict=pd.read_csv('./Daten/to_predict.csv', sep=';',  decimal=',', error_bad_lines=False)


#fill missing data with mean value, dont drop the lines with missing data! 
measures_head = list(measures.head(0))
measures_head.remove('P-KennungAnonym')
measures_head.remove('P-Altersklasse')
for x in measures_head:
    measures[x].fillna(measures[x].mean(), inplace=True)
    predict[x].fillna(predict[x].mean(), inplace=True)





#dtreeOutput = dtree(k=5).complete(measures)
#dtreeOutput = dtree(k=5).complete(predict)


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
Xtest_set =  Xtest_set.values

y = training_set['P-Altersklasse']
y = y.values
#print(len(y.index))

ytest_set = test_set['P-Altersklasse']
ytest_set = ytest_set.values

groups = training_set['P-KennungAnonym']
#print(len(groups.index))

logo = LeaveOneGroupOut()

#Compute CV-Accuracy
desicion_tree(logo, X,y)

from IPython.display import Image 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO

# Compute Generalization Error
dtree = tree.DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=0)
dtree.fit(X, y)
test_set_accuracy = dtree.score(Xtest_set,ytest_set)
print(test_set_accuracy)

 #Visualize Desicion Tree
def visualize_dtree(dtree, data_head):
   dot_data = StringIO()
   class_names = ["20-29", "30-39", "40-49", "50-59", "60-69"]
   export_graphviz(dtree, out_file=dot_data, feature_names=data_head, class_names = class_names, 
                filled=True, rounded=True, special_characters=True)
   graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
   Image(graph.create_png())
   graph.write_png("dree_42.png")

measures_head = list(measures.head(0))
measures_head.remove('P-KennungAnonym')
measures_head.remove('P-Altersklasse')
visualize_dtree(dtree, measures_head)

# Create PDF
#graph.write_pdf("dtree.pdf")

# Create PNG
#graph.write_png("dree.png")






