#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 22:38:06 2021

@author: svillarreal
"""

#explore data, create vectorizers and build SVM and DT models


import csv
import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.feature_extraction
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from nltk.stem.porter import PorterStemmer
STEMMER=PorterStemmer()
from statsmodels.stats.outliers_influence import variance_inflation_factor 

#set nltk's porter stemmer into a function

def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(words) for words in words]
    return words

from nltk.stem import WordNetLemmatizer 
LEMMER = WordNetLemmatizer() 

def MY_LEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [LEMMER.lemmatize(words) for words in words]
    return words   

import string
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick
from wordcloud import WordCloud, STOPWORDS
from tabulate import tabulate #create nice looking table
stopwords = set(STOPWORDS)

MyVect_STEM = CountVectorizer(input = "content", 
                              #analyzer='word',
                              stop_words="english",
                              tokenizer = MY_STEMMER, 
                              lowercase=True
                              )

MyVect_LEM = CountVectorizer(input = "content", 
                              #analyzer='word',
                              stop_words="english",
                              tokenizer = MY_LEMMER, 
                              lowercase=True
                              )

MyVect_IFIDF = TfidfVectorizer(input = "content", 
                              analyzer='word',
                              stop_words="english",
                              lowercase=True, 
                              )

MyVect_IFIDF_STEM = TfidfVectorizer(input = "content", 
                              analyzer='word',
                              stop_words="english",
                              tokenizer=MY_STEMMER,
                              lowercase=True, 
                              )

MyVect_IFIDF_LEM = TfidfVectorizer(input = "content", 
                              analyzer='word',
                              stop_words="english",
                              tokenizer=MY_LEMMER,
                              lowercase=True, 
                              )


MyVect_IFIDF_TF = TfidfVectorizer(input = "content", 
                              analyzer='word',
                              stop_words="english",
                              lowercase=True, 
                              use_idf = False
                              )

MyVect_STEM_ubigram = CountVectorizer(input = "content", 
                              #analyzer='word',
                              stop_words="english",
                              tokenizer = MY_STEMMER, 
                              lowercase=True, 
                              ngram_range = (1,2)
                              )

MyVect_LEM_ubigram = CountVectorizer(input = "content", 
                              #analyzer='word',
                              stop_words="english",
                              tokenizer = MY_LEMMER, 
                              lowercase=True,
                              ngram_range = (1,2)
                              )

MyVect_IFIDF_STEM_ubigram = TfidfVectorizer(input = "content", 
                              analyzer='word',
                              stop_words="english",
                              tokenizer=MY_STEMMER,
                              lowercase=True, 
                              ngram_range = (1,2),
                              min_df = 4
                              )

MyVect_IFIDF_LEM_ubigram = TfidfVectorizer(input = "content", 
                              analyzer='word',
                              stop_words="english",
                              tokenizer=MY_LEMMER,
                              lowercase=True, 
                              ngram_range = (1,2)
                              )



yummly_df = pd.read_csv('/Users/svillarreal/Desktop/Text Mining/project/Yummly_Data.csv')

allLabelsList = yummly_df['cuisine'].tolist()
allRecipesList = yummly_df['ingredients'].tolist()
allRecipesList[0]
#test logic to clean recipes
res = str(allRecipesList[0])[1:-1] 
res = res.replace("'", "")
res = res.replace(',', '')
res

test =re.sub(r'\s+', '-', allRecipesList[0])
test = str(test)[1:-1] 
test = test.replace("'", "")
test = test.replace(',-', ' ')

comb = test+" "+res
comb = comb.split()
comb = " ".join(sorted(set(comb), key=comb.index))

#clean recipe data
newRecipesList = []
for i in range(len(allRecipesList)):
    res = str(allRecipesList[i])[1:-1]
    res = res.replace("'", "")  
    res = res.replace(',', '')
    test =re.sub(r'\s+', '-', allRecipesList[i])
    test = str(test)[1:-1] 
    test = test.replace("'", "")
    test = test.replace(',-', ' ')
    comb = test+" "+res
    comb = comb.split()
    comb = " ".join(sorted(set(comb), key=comb.index))
    newRecipesList.append(comb)
    

##write function that returns True if numbers in string
def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


def textMinDF(vectorizer):
    #fit transform
    x1 =vectorizer.fit_transform(newRecipesList)
    #column names
    
    columnNames1 = vectorizer.get_feature_names()
    
    #create pandas data frame of vectorizer
    builderS = pd.DataFrame(x1.toarray(), columns = columnNames1)
    #for builderS clean data 
    for nextcol in builderS.columns:
        LogResult=Logical_Numbers_Present(nextcol)
        if(LogResult == True):
            builderS = builderS.drop([nextcol], axis=1)
        #next remove any words that are shorter than 3 characters
        elif(len(str(nextcol)) <=3):
            builderS=builderS.drop([nextcol], axis=1)
    #get true/lie label and add it as column on builderS pd dataframe
    builderS["cuisine"]=allLabelsList
    #create local name of final df
    FinalDF=pd.DataFrame()
    #write cleaner builderdf to just created df
    FinalDF= FinalDF.append(builderS)
    #final clean up replace NaN with 0 to make sure models can run
    FinalDF = FinalDF.fillna(0)
    #remove reviews that don't add value as they are blank
    return(FinalDF)
    
#####SELECT VECTORIZER AND CHANGE IT HERE INSIDE textMinDF() FUNCTION######## 
FinalDF_STEM = textMinDF(MyVect_IFIDF_LEM)
#MyVect_IFIDF without stemmer, 79% with sd .004

#FinalDF_STEM = textMinDF(MyVect_IFIDF_TF)

#####################set train and test datasets######
#set test and train model
from sklearn.model_selection import train_test_split
import random as rd

rd.seed(1234)
#create train and test splits
#countvect
trainDF1, testDF1 = train_test_split(FinalDF_STEM, test_size = 0.3)
test1Labels = testDF1["cuisine"]
testDF1 = testDF1.drop(["cuisine"], axis=1)
train1Labels = trainDF1["cuisine"]
trainDF1 = trainDF1.drop(["cuisine"], axis=1)


############################create SVM####################

#create confusion matrix with labels for confusion matrix with model name
def cnfMatrixOutput(cnfMatrix, vectorizerName):
    cnfMatrix =pd.DataFrame(data=cnfMatrix, index=["F", "T"],
                   columns=["F", "T"])
    print(tabulate(cnfMatrix, headers = 'keys', tablefmt = 'psql'),
          (((cnfMatrix.iloc[0,0]+cnfMatrix.iloc[1,1])/cnfMatrix.values.sum())*100).round(0),
          "%",
          vectorizerName)



from sklearn.svm import LinearSVC
###############MyVect_Stem##############
#linear kernel
#sklearn.svm.SVC(C=50, kernel='rbf', verbose=True, gamma="auto")

SVM_Model_1a = LinearSVC(C=1)#LinearSVC(C=.001)#LinearSVC(C=1)
#SVM_Model_1b = LinearSVC(C=.001)
#SVM_Model_1c = LinearSVC(C=10)
#stemming
#C1 = 77%, C.01-> 70%  C10=75% _> sd .002
#lemming
#C1 = 78%, C01 ->70%, C10=76% -? sd .003
#model training
#tfid no lematizer
#C1 78%, sd .004
#tfid with stemmer
#C1 78% sd .0036
#tfidf with lemmer
#C1 78% sd .0033
#TFID without lemmer and with only TF no weights, idf=F
#C1 79% sd .0034

SVM_Model_1a.fit(trainDF1, train1Labels)
#SVM_Model_1b.fit(trainDF1, train1Labels)
#SVM_Model_1c.fit(trainDF1, train1Labels)


#model testing
from sklearn.metrics import confusion_matrix
SVM_matrix_1a = confusion_matrix(test1Labels, SVM_Model_1a.predict(testDF1))#48
print(SVM_matrix_1a)
#SVM_matrix_1b = confusion_matrix(test1Labels, SVM_Model_1b.predict(testDF1))
#SVM_matrix_1c = confusion_matrix(test1Labels, SVM_Model_1c.predict(testDF1))

#prediction
prediction1 = SVM_Model_1a.predict(testDF1)

######look at confusion matrix report#####################
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print(precision_score(test1Labels, prediction1, average=None))
print(recall_score(test1Labels, prediction1, average=None))
from sklearn.metrics import classification_report
target_names = ['japanese',
 'chinese',
 'indian',
 'filipino',
 'mexican',
 'british',
 'vietnamese',
 'thai',
 'korean',
 'jamaican',
 'italian',
 'cajun_creole',
 'moroccan',
 'brazilian',
 'french',
 'spanish',
 'southern_us',
 'greek',
 'russian',
 'irish']
print(classification_report(test1Labels, prediction1, target_names=target_names))

##############plot to see how balanced is the dataset###################
from collections import Counter
#create bar plot for positive and negative 
#make labels a list
res_LabelsToList= FinalDF_STEM['cuisine'].tolist()
#look at counts of label
counts = Counter(res_LabelsToList)
#create labels and values
labels, values = zip(*counts.items())
#sort by top value
indSort = np.argsort(values)[::-1]
#rearrange daata
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]/len(res_LabelsToList) #remoove length if want frequency
                                                        #only 
indexes = np.arange(len(labels))
bar_width = 0.35
# add labels
fig, axs = plt.subplots(1,1)
axs.bar(labels,values)
axs.set_title('Cuisines')
axs.set_xticklabels(target_names, rotation=90)

##################function to check most important features##############
def show_most_and_least_informative_features(vectorizer, clf, class_idx=0, n=10):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[class_idx], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[-n:])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

#print 10 most important features per cousine
for i in range(len(target_names)):
    ##########DON'T FORGET TO CHANGE THE FIRST PARAMETER#################3
    print(show_most_and_least_informative_features(MyVect_IFIDF_TF, SVM_Model_1a, class_idx=i, n=10),
      target_names[i])


##############set up 10 fold cross validation##################3
from sklearn.model_selection import cross_val_score   
from sklearn.model_selection import ShuffleSplit
#count vectorizer with stemmer 
myModelSVM1a_all = LinearSVC(C=1, max_iter=10000)
FinalDF_STEM_noLabel_a =  FinalDF_STEM
FinalDF_STEM_labels_a = FinalDF_STEM_noLabel_a["cuisine"]
FinalDF_STEM_noLabel_a = FinalDF_STEM_noLabel_a.drop(["cuisine"], axis=1)

def matrixtoFrame(frame, algName):
    df = pd.DataFrame(frame)
    df.columns = [algName]
    print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
    print(tabulate(df.describe(), headers = 'keys', tablefmt = 'psql'))
#7, 55%
cv = cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=1717) #1717

matrixtoFrame(cross_val_score(myModelSVM1a_all,FinalDF_STEM_noLabel_a, FinalDF_STEM_labels_a,
                         cv=cv), "CV_STEM_LSVC")#55%


















