# encoding=utf8  

from importlib import reload #doesnt work with python 2.7
#from __future__ import unicode_literals # dont remove this! hande umlaut in plt title
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
import pydotplus
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


measures=pd.read_csv('./Daten/measures_0_feat_deleted.csv',  sep=';', decimal=',', error_bad_lines=False)
predict=pd.read_csv('./Daten/to_predict.csv', sep=';',  decimal=',', error_bad_lines=False)



#fill missing data with mean value, dont drop the lines with missing data! 
measures_head = list(measures.head(0))





measures_head.remove('P-KennungAnonym')
measures_head.remove('P-Altersklasse')
measures_head.remove('P-Geschlecht')

# print number of NaN in columns to calculate % of missing values
null_columns=[]
for x in measures_head:
    if measures[x].isnull().sum() > 0:
       # print(x + str(measures[x].isnull().sum()))
        null_columns.append(x)


for x in measures_head:
    measures[x].fillna(measures[x].mean(), inplace=True)
    predict[x].fillna(predict[x].mean(), inplace=True)

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#measures = scaler.fit_transform(measures)

corr = []
#print correlation btw columns that consist NaN values and class variable
for n in measures_head: 
   if ( -0.2 < measures[n].corr(measures['P-Altersklasse']) < 0.2):
       print ("Low correlation")
       print (n + " " + str(measures[n].corr(measures['P-Altersklasse'])))
   elif ( -1.0 < measures[n].corr(measures['P-Altersklasse']) < -0.4 or  0.4 < measures[n].corr(measures['P-Altersklasse']) < 1.0):
       print ("High correlation")
       print (n + " " + str(measures[n].corr(measures['P-Altersklasse'])))
   #print(measures[['P-Altersklasse',n]].corr())

cmap = cmap=sns.diverging_palette(220, 20, sep=20, as_cmap=True)
# calculate the correlation matrix
corr = measures.corr()

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "8pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '8pt')])
]

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '30px', 'font-size': '8pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())

# plot the heatmap
plt.figure(num=None, figsize=(30, 30), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

plt.show()

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
#print(training_set)


#ID doesn't contain any useful information, so we can just delete this, then set all to X instead of P-Altersklasse column
columns = ['P-KennungAnonym', 'P-Altersklasse']
X_train = training_set.drop(columns, 1)



#set P-Altersklasse column to y
y_train = training_set['P-Altersklasse']


X_test = test_set.drop(columns, 1)
y_test = test_set['P-Altersklasse']
#***************************************************Vergleich Klassenverteilung in Trainings- und +Testmenge (Klassenhistogramme)******************************************************/

#Class diestribution
#sns.distplot(measures['P-Altersklasse'])
#plt.show()

def plot_class_distribution(label_set):
    class_0 = (label_set.loc[(measures["P-Altersklasse"] == "20-29")]).__len__()
    class_1 = (label_set.loc[(measures["P-Altersklasse"] == "30-39")]).__len__()
    class_2 = (label_set.loc[(measures["P-Altersklasse"] == "40-49")]).__len__()
    class_3 = (label_set.loc[(measures["P-Altersklasse"] == "50-59")]).__len__()
    class_4 = (label_set.loc[(measures["P-Altersklasse"] == "60-69")]).__len__()
    classes= [class_0, class_1, class_2, class_3, class_4]

    classes_common = 0
    for c in classes: 
        classes_common = classes_common + c
   # print classes_common
    
    classes_freq = []
    for c in classes:
        c = c/float(classes_common)*100
        classes_freq.append(int(c))
    
    #print classes_freq
    return classes_freq


#Trainingsmenge
#classes_training = plot_class_distribution(y_train)
#x_axes = ["20-29", "30-39", "40-49", "50-59", "60-69"]
#plt.bar(range(len(classes_training)),classes_training, align='center')
#plt.xticks(range(len(classes_training)),x_axes, size='small')
#plt.title("Klassenverteilung in Trainingsmenge")
#plt.xlabel("Altersklasse")
#plt.ylabel("Häufigkeit in %")
#plt1 = plt.show()

#Testmenge
#classes_test = plot_class_distribution(y_test)
#x_axes = ["20-29", "30-39", "40-49", "50-59", "60-69"]
#plt.bar(range(len(classes_test)),classes_test, align='center', color="green")
#plt.xticks(range(len(classes_test)),x_axes, size='small')
#plt.title("Klassenverteilung in Testmenge")
#plt.xlabel("Altersklasse")
#plt.ylabel("Häufigkeit in %")
#plt2 = plt.show()


data_head = list(X_train.head(0))
length = data_head.__len__()

#def statistics(data_head, measures):
 #   for x in  data_head:
  #      index = data_head.index(x)
   #     mean = measures[x].values.mean()
    #    f_max = measures[x].values.max()
    #    f_min = measures[x].values.min()
     #   std = measures[x].values.std()
      #  variance = measures[x].values.var()
      #  print(x + "\n Mean " + str(mean) +
       #         "\n Max " + str(f_max) +
        #        "\n Min" + str(f_min) + 
         #       "\n Standart deviation" + str(std) + 
          #      "\n Variance" + str(variance))

#statistics(data_head, measures)

# -*- coding: utf-8 -*-
def detect_outlier(column, data_frame):
    x_axes= np.arange(0, len(measures[column].index), 1)
    plt.scatter(x_axes, measures[column].values)
    plt.title('2D-Ausreißeranalyse für ' + column)
    plt.xlabel("Spaltenindex in dem Data Frame")
    plt.ylabel(column)
    plt.show()
    plt.savefig('./correlation.pdf')

outlier_analyse = ['L-firstStep','L-abrollwinkel',
                    'L-OKtilt','L-meanROMSpineX','R-firstStep', 
                    'R-meanAmplMidSwing', 'R-aufsetzwinkel', 
                    'S-lastStep','S-meanStride','S-meanStance','S-SwingStride',
                    'S-meanDoubleLegSupport']

#for o in outlier_analyse:
#    detect_outlier(o, measures)


#***************************************************Boxplot wichtiger Merkmale über der Klasse******************************************************/
#import seaborn as sns
#https://stats.stackexchange.com/questions/273108/what-exactly-does-a-boxplot-show?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
# explanation of boxplot 
#https://matplotlib.org/faq/howto_faq.html#how-to-search-examples

#def boxplot(data_head, measures):
 #   sns.set_style("whitegrid")

  #  y_axes= "P-Altersklasse"
   # for x in  data_head[0::4]:
    #    index = data_head.index(x)
     #   if (index == 92):
      #      ax1 = plt.subplot(221)
       #     sns.boxplot(x=x, y=y_axes,data=measures, orient="h", palette="Set2")
            
        #    ax2 = plt.subplot(222, sharey=ax1)
         #   sns.boxplot(x=data_head[index+1], y=y_axes,data=measures, orient="h", palette="Set2")

          #  ax3 = plt.subplot(223, sharey=ax1)
           # sns.boxplot(x=data_head[index+2], y=y_axes,data=measures, orient="h", palette="Set2")
           # plt.show()
        #else: 
         #   ax1 = plt.subplot(221)
          #  sns.boxplot(x=x, y=y_axes,data=measures, orient="h", palette="Set2")
            
           # ax2 = plt.subplot(222, sharey=ax1)
            #sns.boxplot(x=data_head[index+1], y=y_axes,data=measures, orient="h", palette="Set2")

            #ax3 = plt.subplot(223, sharey=ax1)
            #sns.boxplot(x=data_head[index+2], y=y_axes,data=measures, orient="h", palette="Set2")

            #ax3 = plt.subplot(224, sharey=ax1)
            #sns.boxplot(x=data_head[index+3], y=y_axes,data=measures, orient="h", palette="Set2")
            #plt.show()



#boxplot(data_head, measures)






