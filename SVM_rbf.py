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

def svm_rbf(cv, X, y):
    # empty list that will hold cv scores
    svc_acc_train = []
    svc_acc_test = []
    gamma = [x / 10.0 for x in range(1, 20, 2)]
    #print(gamma)
    mean_test = []
    mean_train =[]

    for i in gamma:
        clf = SVC(kernel='rbf', C=1,gamma = i, random_state=0)
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
        mean_train.append(mean_accuracy_clf_train)
        #print("mean accuracy SVC train " + str(i) + " " + str(mean_accuracy_clf_train))
        mean_accuracy_clf_test = accuracy_common_test/count
        mean_test.append(mean_accuracy_clf_test)
        print("mean accuracy SVC train " + str(i)  + " " + str(mean_accuracy_clf_train))
        print("mean accuracy SVC test " + str(i)  + " " + str(mean_accuracy_clf_test))

    plt.scatter(gamma,mean_train,color='red', label="Trainingsmenge")
    plt.scatter(gamma,mean_test,color='blue', label="Testmenge")
    plt.title("Abhähngigkeit der Erfolgsrate von Gamma-Parameter der SVM mit RBF Kernel")
    plt.xlabel("Erfolgsrate")
    plt.ylabel("Gamma-Parameter")
    plt.show()

measures=pd.read_csv('./Daten/measures_feat_deleted.csv',  sep=';', decimal=',', error_bad_lines=False)
predict=pd.read_csv('./Daten/to_predict.csv', sep=';',  decimal=',', error_bad_lines=False)



#fill missing data with mean value, dont drop the lines with missing data! 
measures_head = list(measures.head(0))

measures_head_full = measures_head
print(measures_head)




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
#print ("Number of rows in trainings_set " + len(training_set.index))

#print(training_set)
from sklearn.model_selection import LeaveOneGroupOut
columns = ['P-KennungAnonym', 'P-Altersklasse']
X = training_set.drop(columns, 1)
features = training_set.drop(columns, 1).head(0)
x_axes = X.head(0)
X = X.values

Xtest_set = test_set.drop(columns, 1)
Xtest_set = Xtest_set.values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
Xtest_set = scaler.fit_transform(Xtest_set)

y = training_set['P-Altersklasse']
y = y.values
ytest_set = test_set['P-Altersklasse']
ytest_set = ytest_set.values
#print(len(y.index))

groups = training_set['P-KennungAnonym']
#print(len(groups.index))
logo = LeaveOneGroupOut()

#svm_rbf(logo, X, y)

# Compute Generalization Error
clf = SVC(kernel='rbf', C=1,gamma = 1.1, random_state=0)
clf.fit(X, y)
test_set_accuracy = clf.score(Xtest_set,ytest_set)
print(test_set_accuracy)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#from sklearn.datasets import make_classification
#from sklearn.ensemble import ExtraTreesClassifier

#etc = ExtraTreesClassifier()
#etc = etc.fit(X, y)
#print(etc.feature_importances_)  

#feat_a = []
#imp = []
#importances = etc.feature_importances_
#std = np.std([tree.feature_importances_ for tree in etc.estimators_],
#             axis=0)
#indices = np.argsort(importances)[::-1]


#for feat, importance in zip(features, etc.feature_importances_):
  #  feat_a.append(feat)
 #   imp.append(importance)
#print('feature: {f}, importance: {i}'.format(f=feat, i=importance))

#print(sorted(imp))

#for x in range(10, 101, 10): 
    #if x == 10: 
      #  feat_a_1 = feat_a[:10]
     #   imp_1 = imp[:10]
    #elif x == 100: 
      #  feat_a_1 = feat_a[x-10:95]
     #   imp_1 = imp[x-10:95]
    #else: 
     #   feat_a_1 = feat_a[x-10:x]
      #  imp_1 = imp[x-10:x]
    #plt.bar(range(len(imp_1)),imp_1, align='center', color="green")
    #plt.xticks(range(len(imp_1)),feat_a_1, size='small')
    #plt.ylim(0.0, 0.2)
    #plt.title("Klassenverteilung in Testmenge")
   # plt.xlabel("Altersklasse")
  #  plt.ylabel("Häufigkeit in %")
 #   plt2 = plt.show()




#plt.bar(range(len(etc.feature_importances_)),etc.feature_importances_, align='center', color="green")
#plt.xticks(range(len(etc.feature_importances_)),x_axes, size='small')
#plt.title("Klassenverteilung in Testmenge")
#plt.xlabel("Altersklasse")
#plt.ylabel("Häufigkeit in %")
#plt2 = plt.show()


 
