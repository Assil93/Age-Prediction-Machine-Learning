# encoding=utf8  
#from importlib import reload #doesnt work with python 2.7
from importlib import reload
import sys 
reload(sys)  
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

measures=pd.read_csv('C:/Users/Assil/Desktop/AgePredictionFromGang-wip/Daten/measures.csv',  sep=';', decimal=',', error_bad_lines=False)
predict=pd.read_csv('C:/Users/Assil/Desktop/AgePredictionFromGang-wip/Daten/to_predict.csv', sep=';',  decimal=',', error_bad_lines=False)



#fill missing data with mean value, dont drop the lines with missing data! 
measures_head = list(measures.head(0))
measures_head.remove('P-KennungAnonym')
measures_head.remove('P-Altersklasse')
measures_head.remove('P-Geschlecht')
for x in measures_head:
    measures[x].fillna(measures[x].mean(), inplace=True)
    predict[x].fillna(predict[x].mean(), inplace=True)


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

y = training_set['P-Altersklasse']
y = y.values
#print(len(y.index))

groups = training_set['P-KennungAnonym']
#print(len(groups.index))
logo = LeaveOneGroupOut()

knn = KNeighborsClassifier()


for train, test in logo.split(X, y, groups=groups):
     print("%s %s" % (train, test))
     X_train, X_test = X[train], X[test]
     y_train, y_test = y[train], y[test]
     #print(X_train, X_test, y_train, y_test)
     #print(X_test)
     knn.fit(X_train,y_train)
     print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
     print('Accuracy of K-NN classify_train, y_test = y[train_index], y[test_index]bier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))




#ID doesn't contain any useful information, so we can just delete this, then set all to X instead of P-Altersklasse column
#columns = ['P-KennungAnonym', 'P-Altersklasse']
#X_train = training_set.drop(columns, 1)



#set P-Altersklasse column to y
#y_train = training_set['P-Altersklasse']


#X_test = predict_set.drop(columns, 1)
#y_test = predict_set['P-Altersklasse']


#***************************************************Classification*****************************************************/

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))

#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)

#scaler = MinMaxScaler()
#X_test = scaler.fit_transform(X_test)

svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))

clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

svm = LinearSVC(random_state=0)
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))


for i in list([ 'sigmoid', 'poly' , 'linear', 'rbf']):
    if (i=='poly'):
        clf = SVC(kernel=i, C=1,degree=1, random_state=0)
    elif (i == 'rbf'):
        clf = SVC(kernel=i, C=1,gamma=0.2, random_state=0)
    else:
        clf = SVC(kernel=i, C=1, random_state=0)
    clf = clf.fit(X_train, y_train)
    print("Loop Kernel\n")
    print(i, clf.score(X_test,y_test))



#***************************************************kNN*******************************************************/
clf = neighbors.KNeighborsClassifier(n_neighbors = 5) #default is 5
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("kNN accuracy " +  str(accuracy))


# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbs = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
knn_acc = []
neighbors_k = range(1,30,1)

for k in neighbors_k:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy_knn = knn.score(X_test, y_test)
    print(accuracy_knn, k)
    knn_acc.append(accuracy_knn)

plt.scatter(neighbors_k,knn_acc,color='red')
plt.show()


# changing to misclassification error
#MSE = [1 - x for x in cv_scores]

# determining best k
#optimal_k = neighbs[MSE.index(min(MSE))]
#print "The optimal number of neighbors is %d" % optimal_k

# plot misclassification error vs k
#plt.plot(neighbs, MSE)
#plt.xlabel('Number of Neighbors K')
#plt.ylabel('Misclassification Error')
#plt.show()


#***************************************************Desicion Tree*******************************************************/
#!The Problem of Desicion Tree is overfitting!
#fit the classifier
dtree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=1, min_samples_split=2)
dtree.fit(X_train, y_train)

accuracy_tree = dtree.score(X_test, y_test)
print("Desicion Tree Accuracy " +  str(accuracy_tree))

train_results = []
test_results = []

#max_depth = np.arange(1, 51,1)
#print(max_depth)
#for md in max_depth:
#    dtree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=md, random_state=0)
#    dtree.fit(X_train, y_train)
#    accuracy_tree = dtree.score(X_test, y_test)
#    test_results.append(accuracy_tree)

#plt.scatter(test_results,max_depth,color='red')
#plt.show()

#min_samples_split = np.arange(2, 50, 1)
#for ms in min_samples_split:
#    dtree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10,min_samples_split=ms, random_state=0)
#    dtree.fit(X_train, y_train)
#    accuracy_tree = dtree.score(X_test, y_test)
#    test_results.append(accuracy_tree)

#plt.scatter(test_results,min_samples_split,color='blue')
#plt.show()

#min_samples_leaf = np.arange(1, 50, 1)
#for ms in min_samples_leaf :
#    dtree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10,min_samples_split=2,min_samples_leaf=ms, random_state=0)
#    dtree.fit(X_train, y_train)
#    accuracy_tree = dtree.score(X_test, y_test)
#    test_results.append(accuracy_tree)

#plt.scatter(test_results,min_samples_leaf ,color='blue')
#plt.show()





#Visualize Desicion Tree
#dot_data = StringIO()
#class_names = ["20-29", "30-39", "40-49", "50-59", "60-69"]
#export_graphviz(dtree, out_file=dot_data, feature_names=data_head, class_names = class_names, 
                #filled=True, rounded=True,
               # special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())



# Create PDF
#graph.write_pdf("dtree.pdf")

# Create PNG
#graph.write_png("dree.png")



# predict the age of test persons
#X_predict = np.array(predict)
#predicted = dtree.predict(X_predict)
#print(dtree.predict(X_predict))

#prediction = pd.DataFrame(dtree.predict(X_predict), columns=['P-Altersklasse-Predicted']).to_csv('C:/Users/Sapsarap/Desktop/Darya/Semester 2/Data Mining/DatenSS2018/to_predict_new2.csv', sep=";")
