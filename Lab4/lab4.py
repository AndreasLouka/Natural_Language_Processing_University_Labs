import numpy as np
import random, os, glob, time, nltk, re
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from random import shuffle
from collections import Counter
from itertools import product
#np.set_printoptions(threshold=np.nan)
import operator
import time


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
    the_dict = dict()
    counter = 0

    for sentence in X:
        features = Counter()

        for word_tag in sentence:
            features[word_tag[0]+"_"+word_tag[1]] += 1
        
        for i,tag_tag in enumerate(sentence):
            if (i <len(sentence)-1):
                features[sentence[i][1]+"_"+sentence[i+1][1]] += 1
            else:
                features[sentence[i][1]+"_"+"END"] +=1

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


#function to create all possible word_tag combinations for 1 sentence
def combinations (X,Y):
    for pair in product(Y,repeat=len(X)):
        yield ' '.join('%s_%s'%t for t in zip(X,pair))

#function to create all possible tag_tag combinations for 1 sentence
#(word_tag combinations created using the function 'combinations' were used to extract the tag_tag combinations,
# by extracting the tag_tag combinations from the word_tag combinations in a similar way as used to extract the 
# tag_tag combinations from the original dataset, in the function 'features')
def combinations_tag_tag (comb):
    features = Counter()
    spl = comb.split(" ")
    word_tag_all = []

    for word_tag in spl:
        word_tag_all.append(word_tag)

    count = 0
    for i in range(len(word_tag_all)):
        if (i < len(word_tag_all)-1):
            if (len(word_tag_all[i].split("_")) > 1):
                features[word_tag_all[i].split("_")[1]+"_"+word_tag_all[i+1].split("_")[1]] += 1
        else:
            if (len(word_tag_all[i].split("_")) > 1):
                features[word_tag_all[i].split("_")[1]+"_"+"END"] +=1
    count += 1

    return(features)

#training function
def training(X):
    #create a list with all the POS tags using the 'tags' function
    all_tags = tags(X)
    
    #create variables: 
    weights = dict()
    sentence_word_list = []


    
    for key in X.keys():
    #First of all, create the weights dictionary and initialise all weights to 0
        for value in X[key].keys():
            weights[value] = 0   

    for i in range(5):  
        #start loop for every key (sentence) in the dictionary:
        for key in X.keys():
        
            #Extract all the words for each sentence, by using the 'sentence_words' function
            sentence_word_list = sentence_words(X[key], all_tags)
            
            # #Create all the word_tag combinations for each sentence, using the function 'combinations' 
            sentence_word_tag_combinations = combinations(sentence_word_list, all_tags) 
            
            

            # #create all the possible tag_tag combinations using the 'combinations_tag_tag' function
            #and perform the argmax:
            current_max = -1
            maxcomb = None
            for comb in sentence_word_tag_combinations:
                phi = 0
                sentence_tag_tag_combinations = combinations_tag_tag(comb)

                for word_tag in comb.split(" "):
                    if (word_tag in weights.keys()):
                        phi += weights[word_tag]

                for tag_tag in sentence_tag_tag_combinations:
                    if (tag_tag in weights.keys()):
                        phi += sentence_tag_tag_combinations[tag_tag] * weights[tag_tag]

                if (phi > current_max):
                    current_max = phi
                    maxcomb = comb

            max_sentence_tag_tag_combinations = combinations_tag_tag(maxcomb)

            for word_tag in maxcomb.split(" "):
                if (word_tag not in X[key].keys()):
                    if (word_tag in weights.keys()):
                        weights[word_tag] -= 1

            for word_tag in X[key].keys():
                if (word_tag not in maxcomb.split(" ")):
                    if word_tag in weights.keys():
                        weights[word_tag] += 1



            for tag_tag in max_sentence_tag_tag_combinations:
                if (tag_tag not in X[key].keys()):
                    if (tag_tag in weights.keys()):
                        weights[tag_tag] -=1

            for tag_tag in X[key].keys():
                if (tag_tag not in max_sentence_tag_tag_combinations):
                    if tag_tag in weights.keys():
                        weights[tag_tag] += 1

            

    #AVERAGING
    new_dict = dict()
    for key in weights.keys():
        new_dict[key] = weights[key] / 5


    return(new_dict)
    #return(weights)




#testing function:
def testing(X, weights):
    correct_counter = 0
    word_counter = 0

    #create a list with all the POS tags using the 'tags' function
    all_tags = tags(X)
    
        #start loop for every key (sentence) in the dictionary:
    for key in X.keys():
    
        #Extract all the words for each sentence, by using the 'sentence_words' function
        sentence_word_list = sentence_words(X[key], all_tags)
        
        # #Create all the word_tag combinations for each sentence, using the function 'combinations' 
        sentence_word_tag_combinations = combinations(sentence_word_list, all_tags) 
        
        # #create all the possible tag_tag combinations using the 'combinations_tag_tag' function
        current_max = -1
        maxcomb = None
        for comb in sentence_word_tag_combinations:
            phi = 0
            sentence_tag_tag_combinations = combinations_tag_tag(comb)

            for word_tag in comb.split(" "):
                if (word_tag in weights.keys()):
                    phi += weights[word_tag]

            for tag_tag in sentence_tag_tag_combinations:
                if (tag_tag in weights.keys()):
                    phi += sentence_tag_tag_combinations[tag_tag] * weights[tag_tag]

            if (phi > current_max):
                current_max = phi
                maxcomb = comb

        max_sentence_tag_tag_combinations = combinations_tag_tag(maxcomb)

        
        for word_tag in maxcomb.split(" "):
            word_counter += 1
            if (word_tag in X[key].keys()):
                correct_counter += 1

        for tag_tag in max_sentence_tag_tag_combinations:
            word_counter += 1
            if (tag_tag in X[key].keys()):
                correct_counter +=1


    #return accuracy (calculated by dividing the correct predictions with the total number of predictions):
    return(correct_counter / word_counter)



###################################################################################################################

#Load data (the sentences) from the Brown corpus up to length 5:
sents = nltk.corpus.brown.tagged_sents(tagset="universal")
sentsMax5 = []
for sent in sents:
    if len(sent)<5:
        sentsMax5.append(sent)

###############

#Split data to training and testing using "split_data_to_train_and_test" function:
#all data = 3235 sentences | training data set = 2500 sentences | testing data set = 735 sentences
training_set, testing_set = split_data_to_train_and_test(sentsMax5)

###############

start_split = time.time()
#Create trainining and testing dictionaries
train_features_dict = features(training_set)
test_features_dict = features(testing_set)
end_split = time.time()

###############

start_training = time.time()
#Create weights dictionary with the 'training' function
weights = training(train_features_dict)
end_training = time.time()

###############

start_testing = time.time()
#Accuracy from the 'testing' function
accuracy = testing(test_features_dict, weights)
print("\nThe accuracy is: ", accuracy)
end_testing = time.time()

###############

#Print highest-weighted features
# print(sorted(weights, key=weights.get, reverse=True)[:10])
highest_weights = dict(sorted(weights.items(), key=operator.itemgetter(1), reverse=True)[:10])
print("\nThe highest-Weighted Features are: ", highest_weights)

###############
print("\nExecution time for training (argmax) is: ", end_training - start_training)
print("\nExecution time for splitting data is: ", end_split - start_split)
print("\nExecution time for testing is: ", end_testing - start_testing)





