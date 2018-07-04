# -*- coding: utf-8 -*-
"""

Written by Abiola Victor Obamuyide (obabiola@gmail.com)
Adapted by Andreas Vlachos (a.vlachos@sheffield.ac.uk)

"""
import numpy as np;
import re
from os import listdir
from os.path import isfile, join
from collections import Counter,defaultdict
import random


    
    
filepath_pos = "review_polarity/txt_sentoken/pos/"
filepath_neg = "review_polarity/txt_sentoken/neg/"

filenames_pos = sorted([filepath_pos+f for f in listdir(filepath_pos) if isfile(join(filepath_pos,f))])
filenames_neg = sorted([filepath_neg+f for f in listdir(filepath_neg) if isfile(join(filepath_neg,f))])

train_filenames_pos = filenames_pos[:800]
train_filenames_neg = filenames_neg[:800]

test_filenames_pos = filenames_pos[800:]
test_filenames_neg = filenames_neg[800:]
#
train_filenames = train_filenames_pos + train_filenames_neg
test_filenames = test_filenames_pos + test_filenames_neg

classes = [1,-1]


def get_data():
    data = []
    new_data = []
    for filename in train_filenames_pos:
        word_list = []
        with open(filename) as fh:
            for line in fh.readlines():
                word_list.extend(re.sub("[^\w']"," ",line).split())
        data.append((1,Counter(word_list)))
        
    for filename in train_filenames_neg:
        word_list = []
        with open(filename) as fh:
            for line in fh.readlines():
                word_list.extend(re.sub("[^\w']"," ",line).split())
        data.append((-1,Counter(word_list)))
    
    for label,feature_dict in data :   
        words = feature_dict.keys()
        values = feature_dict.values()
        feature_dict = Counter(dict(zip(words,values)))
        new_data.append((label,feature_dict))
        
    return new_data

    
    
    
    
def get_test_data():
    data = []
    new_data = []
    for filename in test_filenames_pos:
        word_list = []
        with open(filename) as fh:
            for line in fh.readlines():
                word_list.extend(re.sub("[^\w']"," ",line).split())
        data.append((1,Counter(word_list)))
        
    for filename in test_filenames_neg:
        word_list = []
        with open(filename) as fh:
            for line in fh.readlines():
                word_list.extend(re.sub("[^\w']"," ",line).split())
        data.append((-1,Counter(word_list)))
    for label,feature_dict in data:    
        words = feature_dict.keys()
        values = feature_dict.values()
        feature_dict = Counter(dict(zip(words,values)))
        new_data.append((label,feature_dict))
    return new_data


def populate_dictionary():
    #get the features set
    dictionary_list = []
    for filename in train_filenames:
        with open(filename) as fh:
            for line in fh.readlines():
                word_list = re.sub("[^\w]"," ",line).split()
                dictionary_list += word_list
    return set(dictionary_list)



def train(data,features,epochs=10):
    weights = {}
    averaged_weights = {}
    for c in classes:
        weights[c] = np.zeros(len(features))
        averaged_weights[c] = np.zeros(len(features))
    for e in range(epochs):
        np.random.shuffle(data) 
        for label,feature_dict in data:
            feature_list = [feature_dict[i] for i in features]
            feature_vector = np.array(feature_list,dtype=float)
            arg_max,predicted_class = -np.Infinity,classes[0]
            
            for c in classes:
                activation = np.dot(feature_vector,weights[c])
                if activation >= arg_max:
                    arg_max,predicted_class = activation,c
                    #print arg_max,predicted_class
            
            
            if not(predicted_class == label):
                weights[label] += feature_vector
                weights[predicted_class] -= feature_vector
    return weights
            
    
    
def test(data_test,weights,features):
    results = []
    for label,feature_dict in data_test:
        feature_list = [feature_dict[i] for i in features]
        feature_vector = np.array(feature_list,dtype=float)
        arg_max,predicted_class = -np.Infinity,classes[0]
        
        for c in classes:
            activation = np.dot(feature_vector,weights[c])
            if activation >= arg_max:
                arg_max,predicted_class = activation,c
                
        #print label,predicted_class
        if(predicted_class == label):
            results.append(1)
        else:
            results.append(0)
        
    accuracy = np.sum(np.array(results,dtype=float))/float(len(results))
    return accuracy

    
    

if __name__ == "__main__":
    training_data = get_data()
    #print(training_data[0])
    test_data = get_test_data()           
    features = populate_dictionary()
    #print(len(features))
    np.random.seed(5)
    weights = train(training_data,features, epochs=10)
    accuracy = test(test_data,weights,features)
    print(accuracy)
    
