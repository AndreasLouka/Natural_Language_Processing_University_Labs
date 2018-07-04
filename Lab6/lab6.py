"""\
------------------------------------------------------------
USE: python <PROGNAME> (options) file1...fileN
OPTIONS:

    -v : use viterbi
    
        >> python lab6.py -v


    -b : use beam search

        >> python lab6.py -b
------------------------------------------------------------

"""
import numpy as np
import random, os, glob, time, nltk, re
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from random import shuffle
from collections import Counter, OrderedDict
from itertools import product
#np.set_printoptions(threshold=np.nan)
import operator
import time
import getopt, sys

random.seed(8122547)
R = random.random()

#function to split dataset to training and testing:
#all data = 3235 sentences | training data set = 2500 sentences | testing data set = 735 sentences
def split_data_to_train_and_test(X): 
    training = []
    testing = []
    counter = 0
    for i in range(len(X)):
        if (counter < 2500):
            training.append(X[counter])
            counter += 1
        else:
            testing.append(X[counter])
            counter +=1
    return(training, testing)


#function to extract features from dataset (both word_tag and tag_nextTag)
def features(X):
    the_dict = OrderedDict()
    counter = 0

    for sentence in X:
        features = Counter()

        for word_tag in sentence:
            features[word_tag[0]+"_"+word_tag[1]] += 1
        
        for i,tag_tag in enumerate(sentence):
            if (i > 0) :
                features[sentence[i][1]+"_"+sentence[i-1][1]] += 1
            else:
                features[sentence[i][1]+"_"+"START"] +=1

        the_dict[counter] = features 
        counter += 1

    return(the_dict)

#function to get all possible POS tags (total tags = 13)
def tags(dictionary):
    all_tags = []
    for key in dictionary.keys():
        for tag in dictionary[key].keys():
            split_word_tag = tag.split("_")
            if (split_word_tag[1] not in all_tags):
                all_tags.append(split_word_tag[1])
    return(all_tags)


#function to extract all the words from 1 sentence
def sentence_words(sentence, all_tags):
    words_in_sentence = []
    for word_tag in sentence:
        split_word_tag = word_tag.split("_")
        if (split_word_tag[0] not in all_tags):
            words_in_sentence.append(split_word_tag[0])

    return(words_in_sentence)


#training function
def training(dictionary, arguments):
    #create a list with all the POS tags using the 'tags' function
    all_tags = tags(dictionary)   

    #create weights dictionary (initialize values to 0)
    weights = OrderedDict()
    for key in dictionary.keys():
        for value in dictionary[key].keys():
            if value not in weights.keys():
                weights[value] = 0

    argmax_prediction = []
    if arguments == ['lab6.py', '-v']:
        Argmax = viterbi
    if arguments == ['lab6.py', '-b']:
        Argmax = beam  

    for n in range(0, 5): 
        print('Iteration: ' + str(n))
        for key in dictionary.keys():
            argmax_prediction = Argmax(dictionary[key], weights, all_tags)

            for i in argmax_prediction:
                if i not in dictionary[key]:
                    if i in weights.keys():
                        weights[i] -= 1
                else:
                    if i in weights.keys():
                        weights[i] += 1
        #AVERAGING
        new_dict = OrderedDict()
        for key in weights.keys():
            new_dict[key] = weights[key] / 5

    return(new_dict)


def dot (sentence, w):
    result = 0
    for feature in sentence.keys():
        if feature in w.keys():
            result+= sentence[feature] * w[feature]
    return result


def viterbi(sentence, weights, tags):
    all_words = sentence_words(sentence, tags)
    length_words = len(all_words)
    length_tags = len(tags)

    V = np.zeros((length_tags, length_words))
    B = []
    C = []
    D = []
    E = []
    sequence_words = []
    #word_Tag
    if length_words > 1:
        for i in range(0, length_words):
            for j in range(0, length_tags):
                word_tag = all_words[i]+"_"+tags[j]
                V[j][i] = dot(sentence, weights)
            B.append(all_words[i]+"_"+tags[np.argmax(V[:][i])])
    #tag_Tag
        for i in range(len(B)):
            if i == 0:
                C.append(B[i].split("_")[1]+"_"+"START")
            else:
                C.append(B[i].split("_")[1]+"_"+B[i-1].split("_")[1])
   
    D = B + C
    return(D)


def beam(sentence, weights, tags):

    Beam_size = 13
    Beam_search =[]
    all_words = sentence_words(sentence, tags)
    length_words = len(all_words)
    length_tags = len(tags)
  
    V = np.zeros((length_tags, length_words))
    B = []
    C = []
    D = []
    sequence_words = []
    #word_Tag
    if length_words > 1:
        for i in range(0, length_words):
            for j in range(0, Beam_size):
                word_tag = all_words[i]+"_"+tags[j]
                V[j][i] = dot(sentence, weights)
            B.append(all_words[i]+"_"+tags[np.argmax(V[:][i])])
    #tag_Tag
        for i in range(len(B)):
            if i == 0:
                C.append(B[i].split("_")[1]+"_"+"START")
            else:
                C.append(B[i].split("_")[1]+"_"+B[i-1].split("_")[1])
    D = B + C
    return(D)

def testing(dictionary, weights, arguments):
    #create a list with all the POS tags using the 'tags' function
    all_tags = tags(dictionary)

    all_counter = 0
    correct_counter =0

    argmax_prediction = []
    if arguments == ['lab6.py', '-v']:
        Argmax = viterbi
    if arguments == ['lab6.py', '-b']:
        Argmax = beam  

    for key in dictionary.keys():
        argmax_prediction = Argmax(dictionary[key], weights, all_tags)
        all_counter += len(argmax_prediction)-1 
       
        for i in argmax_prediction:
            if i in weights.keys():
                correct_counter +=1
    return(correct_counter / all_counter)

    
###################################################################################################################
#MAIN:
'__main__'
##############

#Load data (the sentences) from the Brown corpus up to length 5:
sents = nltk.corpus.brown.tagged_sents(tagset="universal")
sentsMax5 = []
for sent in sents:
    if len(sent)<5:
        sentsMax5.append(sent)

###############

#Commandline arguments
arguments = sys.argv
##############

#Split data to training and testing using "split_data_to_train_and_test" function:
#all data = 3235 sentences | training data set = 2500 sentences | testing data set = 735 sentences
training_set, testing_set = split_data_to_train_and_test(sentsMax5)

###############
start_training = time.time()
#Create trainining and testing dictionaries
train_features_dict = features(training_set)
test_features_dict = features(testing_set)
end_training = time.time()

##############

weights = training(train_features_dict, arguments)

# ###############

accuracy = testing(test_features_dict, weights, arguments)
print("Accuracy :", accuracy)

##############
#print("\nExecution time for argmax is: ", end_training - start_training)








