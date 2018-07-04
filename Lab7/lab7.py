p"""\
------------------------------------------------------------
USE: python <PROGNAME> (options) file1...fileN
OPTIONS:

    -1 : use only bog_of_words
    
        >> python lab7.py -1

    -2 : use only word_vectors

        >> python lab7.py -2

    -3 : use both

        >> python lab7.py -3
------------------------------------------------------------

"""



import gensim, logging
from gensim import models
from gensim.models.keyedvectors import KeyedVectors
import time, sys
import numpy as np
import random, os, glob, time, nltk, re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

 
arguments = sys.argv

np.random.seed(423215)


# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)


# similarity_woman = model.most_similar(positive=['woman'], negative=['man'])
# similarity_Leo_Messi = model.most_similar(positive=['Leo_Messi'])
# similarity_France = model.most_similar(positive=['France'])
# similarity_Football = model.most_similar(positive=['Football'])
# similarity_very = model.most_similar(positive=['very'])
# similarity_studying = model.most_similar(positive=['Studying'])
# similarity_sweet = model.most_similar(positive=['sweet'])
# similarity_hello = model.most_similar(positive=['hello'])
# similarity_the = model.most_similar(positive=['the'])
# similarity_tree = model.most_similar(positive=['tree'])


# print("\n Similarity Woman", similarity_woman)
# print("\n Similarity Leo", similarity_Leo_Messi)
# print("\n Similarity France", similarity_France)
# print("\n Similarity Football", similarity_Football)
# print("\n Similarity very", similarity_very)
# print("\n Similarity Studying", similarity_studying)
# print("\n Similarity sweet", similarity_sweet)
# print("\n Similarity hello", similarity_hello)
# print("\n Similarity the", similarity_the)
# print("\n Similarity tree", similarity_tree)

#Function to get the summation of the word vector representations:
def word_vectors(directory, model):    

    #Initialise variables
    word_vector_train = np.zeros(shape = (800, 300))
    word_vector_test = np.zeros(shape = (200, 300))

    
    #Create variable for stopwords (using stopwords from ntlk.corpus)
    cachedStopWords = stopwords.words("english")

    counter = 0
    test_counter = 0

    for file in os.listdir(directory):
        list_bag_of_words = []

        if file.endswith(".txt"): 
        #extend the directory to the subdirectory:
                file = directory + file

                with open (file, 'r') as opened_file:
                    for line in opened_file:
                        #tokenise using the nlkt.word_tokenize function:
                        tokens = nltk.word_tokenize(line)

                        if counter < 800:
                            for token in tokens:
                                #remove non alhabetic characters using a regular expression:
                                token = re.sub("[^A-Za-z]+", "", token)
                                token = token.lower()

                                #Stop-word removal using the english nltk corpus:
                                if (token not in cachedStopWords):
                                    if token in model.vocab:
                                        word = model[token]

                                #SUMMATION:
                                        word_vector_train[counter][:] += word
                        if 800 <= counter < 1000:
                            
                            for token in tokens:
                                token = re.sub("[^A-Za-z]+", "", token)
                                token = token.lower()

                                if (token not in cachedStopWords):
                                    if token in model.vocab:
                                        word = model[token]

                                        word_vector_test[counter-800][:] += word

        counter += 1
    return(word_vector_train, word_vector_test)

#function to create the bag of words from the documents
def bag_of_words(directory):    
    #create arrays to store the training and testing documents processed
    train_array = []
    test_array = []

    #Initialise counter
    counter = 0
    #Create variable for stopwords (using stopwords from ntlk.corpus)
    cachedStopWords = stopwords.words("english")

    for file in os.listdir(directory):
        list_bag_of_words = []

        if file.endswith(".txt"): 
            #extend the directory to the subdirectory:
            file = directory + file
            
            with open (file, 'r') as opened_file:
                for line in opened_file:
                    #tokenise using the nlkt.word_tokenize function:
                    tokens = nltk.word_tokenize(line)
                    for token in tokens:
                        #remove non alhabetic characters using a regular expression:
                        token = re.sub("[^A-Za-z]+", "", token)
                        token = token.lower()
                        #Stop-word removal using the english nltk corpus:
                        if (token not in cachedStopWords):
                            list_bag_of_words.append(token)

                #Recreate the documents after preprocessing is complete:
                joined = (" ".join(list_bag_of_words))

                #append the first 800 documents to the train array and the last 200 to the test array:
                if (counter <= 800):
                    train_array.append(joined)
                if 800 < counter <= 1000:
                    test_array.append(joined)
                
        counter += 1   

    return(train_array, test_array)




#training function:
def train(X):
    #create empty arrays the same shape as the input array, with one less column
    #(because the last column contains the sign (1 = positive, -1 = negative))
    w_pos= np.zeros(shape = X[0, :-1].shape) 
    w_neg = np.zeros(shape = X[0, :-1].shape)

    #variables for AVERAGING:
    counter_averaging = 1
    w_pos_array = []
    w_neg_array = []

    

    #MULTIPLE PASSES:
    for k in range(100):

        #SUFFLING WITH FIXED RANDOM SEED:
        np.random.shuffle(X)
        #training:
        for i in range(0, len(X)):

            y_pos = np.dot(w_pos, X[i][:-1])
            y_neg = np.dot(w_neg, X[i][:-1])
            
            if (y_pos >= y_neg):
                y_prediction = 1
            else:
                y_prediction = -1

            #Check if predition matches the original weight:
            if np.sign(X[i][-1] * y_prediction) != 1:
            #if y_prediction == -1:

                #if prediction is wrong and the original weight is positive (1), then benefit positive weights
                #and penalise negative weights:
                if (X[i][-1] == 1):
                    w_pos += X[i][:-1]
                    w_neg -= X[i][:-1]
                #if prediction is wrong and the original weight is negative (-1), then benefit negative weights
                #and penalise positive weights:
                else:
                    w_pos -= X[i][:-1]
                    w_neg += X[i][:-1]

    #For AVERAGING:
        w_pos_array.append(w_pos)
        w_neg_array.append(w_neg)
        counter_averaging += 1

    #AVERAGING:
    final_w_pos = np.sum(w_pos_array, axis = 0) / counter_averaging
    final_w_neg = np.sum(w_neg_array, axis = 0) / counter_averaging
    
    return (final_w_pos, final_w_neg)
    




#testing function:
def testing(X, w_pos, w_neg):
    correct_counter = 0
    y_prediction = 0
    np.random.shuffle(X)
    for i in range(0, len(X)):
        y_pos = np.dot(w_pos, X[i][:-1])
        y_neg = np.dot(w_neg, X[i][:-1])

        if (y_pos >= y_neg):
            y_prediction = 1
        else:
            y_prediction = -1

        if np.sign(X[i][-1] * y_prediction) == 1:
        #if y_prediction == 1:

            correct_counter += 1

    #return accuracy (calculated by dividing the correct predictions with the total number of predictions):
    return(correct_counter / len(X))




########################################################################################################################


#create directories for positive and negative documents:
positive = "./pos/"
negative = "./neg/"




###############
#Using only normal Bag_Of_Words approachs (one-hot encodings)
if arguments == ['lab7.py', '-1']:
    ###############

    #call the bag_of_words function with the appropriate directories and create variables for both positive and negative, training/testing bag_of_words:
    bag_pos_train, bag_pos_test = bag_of_words(positive)
    bag_neg_train, bag_neg_test = bag_of_words(negative)

    ###############

    #VECTORIZE THE BAG_OG_WORDS_ARRAYS:

    #create Vectorizers to vectorise the arrays of bag_of_words, using the CountVectorizer imported from sklearn:
    train_vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 50000) 
    test_vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 50000) 
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of strings.
    total_training_Vectorized = train_vectorizer.fit_transform(bag_pos_train + bag_neg_train).toarray()

    #Assign the vocabulary of the test vectorizer to be the same as the vocabulary of the train vectorizer:
    test_vectorizer.vocabulary = train_vectorizer.get_feature_names()

    total_testing_Vectorized = test_vectorizer.fit_transform(bag_pos_test + bag_neg_test).toarray()

    ###############

    #Create vectors with signs (1 = positive, -1=negative) for reviews, which will be combined with the vectorized arrays:
    train_signs = np.empty(shape = (1600, 1))
    train_signs[0:800,0] = 1
    train_signs[800:1600, 0] = -1

    test_signs = np.empty(shape = (400, 1))
    test_signs[0:200,0] = 1
    test_signs[200:400, 0] = -1

    ###############

    #Combine sign vectors with the vectorized bag_of_words arrays (signs will be the last column of the matrix):
    combined_trainData_with_signs = np.concatenate((total_training_Vectorized, train_signs), axis = 1)
    combined_testData_with_signs = np.concatenate((total_testing_Vectorized, test_signs), axis = 1)

    ###############

    #Call the train function (using the combined training array with signs) and create variables for positive and negative weights:
    weights_positive, weights_negative = train(combined_trainData_with_signs)

    #Call the testing function (using the combined testing array with signs):
    total_accuracy = testing(combined_testData_with_signs, weights_positive, weights_negative)
    print("\nAccuracy: {}".format(total_accuracy))






#Using only Word Vector representation:
if arguments == ['lab7.py', '-2']:
    ###############

    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

    ###############


    #Create word_Vector summation using 'word_vector' function:
    word_Vector_pos_train, word_Vector_pos_test = word_vectors(positive, model)
    word_Vector_neg_train, word_Vector_neg_test = word_vectors(negative, model)

    word_vector_all_train = np.concatenate((word_Vector_pos_train, word_Vector_neg_train), axis = 0)
    word_vector_all_test = np.concatenate((word_Vector_pos_test, word_Vector_neg_test), axis = 0)
    

    #Create vectors with signs (1 = positive, -1=negative) for reviews, which will be combined with the vectorized arrays:
    train_signs = np.empty(shape = (1600, 1))
    train_signs[0:800,0] = 1
    train_signs[800:1600, 0] = -1

    test_signs = np.empty(shape = (400, 1))
    test_signs[0:200,0] = 1
    test_signs[200:400, 0] = -1

    ###############

    #Combine sign vectors with the vectorized bag_of_words arrays (signs will be the last column of the matrix):
    combined_trainData_with_signs = np.concatenate((word_vector_all_train, train_signs), axis = 1)
    combined_testData_with_signs = np.concatenate((word_vector_all_test, test_signs), axis = 1)

    ###############

    #Call the train function (using the combined training array with signs) and create variables for positive and negative weights:
    weights_positive, weights_negative = train(combined_trainData_with_signs)

    #Call the testing function (using the combined testing array with signs):
    total_accuracy = testing(combined_testData_with_signs, weights_positive, weights_negative)
    print("\nAccuracy: {}".format(total_accuracy))





#Combining together the one-hot encodings with the vector representations:
if arguments == ['lab7_1.py', '-3']:
    
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

    ###############

    #Create word_Vector summation using 'word_vector' function:
    word_Vector_pos_train, word_Vector_pos_test = word_vectors(positive, model)
    word_Vector_neg_train, word_Vector_neg_test = word_vectors(negative, model)

    word_vector_all_train = np.concatenate((word_Vector_pos_train, word_Vector_neg_train), axis = 0)
    word_vector_all_test = np.concatenate((word_Vector_pos_test, word_Vector_neg_test), axis = 0)

    #call the bag_of_words function with the appropriate directories and create variables for both positive and negative, training/testing bag_of_words:
    bag_pos_train, bag_pos_test = bag_of_words(positive)
    bag_neg_train, bag_neg_test = bag_of_words(negative)
    ###############

    #VECTORIZE THE BAG_OG_WORDS_ARRAYS:

    #create Vectorizers to vectorise the arrays of bag_of_words, using the CountVectorizer imported from sklearn:
    train_vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 50000) 
    test_vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 50000) 

    
    total_training_Vectorized = train_vectorizer.fit_transform(bag_pos_train + bag_neg_train).toarray()

    #Assign the vocabulary of the test vectorizer to be the same as the vocabulary of the train vectorizer:
    test_vectorizer.vocabulary = train_vectorizer.get_feature_names()

    total_testing_Vectorized = test_vectorizer.fit_transform(bag_pos_test + bag_neg_test).toarray()

    ###############

    #Create vectors with signs (1 = positive, -1=negative) for reviews, which will be combined with the vectorized arrays:
    train_signs = np.empty(shape = (1600, 1))
    train_signs[0:800,0] = 1
    train_signs[800:1600, 0] = -1

    test_signs = np.empty(shape = (400, 1))
    test_signs[0:200,0] = 1
    test_signs[200:400, 0] = -1

    ###############
    #Combine sign vectors with the vectorized bag_of_words arrays and the word_vectors:
    combined_trainData_with_signs = np.concatenate((total_training_Vectorized, word_vector_all_train, train_signs), axis = 1)
    combined_testData_with_signs = np.concatenate((total_testing_Vectorized, word_vector_all_test, test_signs), axis = 1)
    ###############

    #Call the train function (using the combined training array with signs) and create variables for positive and negative weights:
    weights_positive, weights_negative = train(combined_trainData_with_signs)

    #Call the testing function (using the combined testing array with signs):
    total_accuracy = testing(combined_testData_with_signs, weights_positive, weights_negative)
    print("\nAccuracy: {}".format(total_accuracy))



