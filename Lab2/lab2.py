import numpy as np
import random, os, glob, time, nltk, re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


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
            c
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
                if (counter < 800):
                    train_array.append(joined)
                else:
                    test_array.append(joined)
                
                counter += 1   

    return(train_array, test_array)




#training function:
def train(X):
    #create empty arrays the same shape as the input array, with one less column
    #(because the last column contains the sign (1 = positive, -1 = negative))
    w_pos= np.zeros(shape = X[0, :-1].shape) 
    w_neg = np.zeros(shape = X[0, :-1].shape)

    #create empty array to store training accuracy
    accuracy_counter_vector = []

    #variables for AVERAGING:
    counter_averaging = 1
    w_pos_array = []
    w_neg_array = []

    #MULTIPLE PASSES:
    for k in range(100):
        #reset training accuracy counter for each iteration:
        training_accuracy_counter = 0

        #SUFFLING WITH FIXED RANDOM SEED:
        random.seed(42)
        #random.shuffle(X)
        #training:
        for i in range(len(X)):

            y_pos = np.dot(w_pos, X[i][:-1])
            y_neg = np.dot(w_neg, X[i][:-1])
            
            if (y_pos >= y_neg):
                y_prediction = 1
            else:
                y_prediction = -1

            #Check if predition matches the original weight:
            if np.sign(X[i][-1] * y_prediction) != 1:
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

            #If prediction matches the original weight, then increment the training accuracy counter:
            else:
                training_accuracy_counter += 1

        #store the accuracy (calculated by dividing the correct predictions with the total predictions):
        accuracy_counter_vector.append(training_accuracy_counter / len(X))

    #For AVERAGING:
        w_pos_array.append(w_pos)
        w_neg_array.append(w_neg)
        counter_averaging += 1

    #AVERAGING:
    final_w_pos = np.sum(w_pos_array, axis = 0) / counter_averaging
    final_w_neg = np.sum(w_neg_array, axis = 0) / counter_averaging


    #plot the accuracy:
    plt.plot(accuracy_counter_vector)
    plt.title("Training Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    #plt.show()

    
    return (final_w_pos, final_w_neg)
    




#testing function:
def testing(X, w_pos, w_neg):
    correct_counter = 0

    for i in range(len(X)):
        y_pos = np.dot(w_pos, X[i][:-1])
        y_neg = np.dot(w_neg, X[i][:-1])

        if (y_pos >= y_neg):
            y_prediction = 1
        else:
            y_prediction = -1

        if np.sign(X[i][-1] * y_prediction) == 1:
            correct_counter += 1

    #return accuracy (calculated by dividing the correct predictions with the total number of predictions):
    return(correct_counter / len(X))








#create directories for positive and negative documents:
positive = "./pos/"
negative = "./neg/"

#call the bag_of_words function with the appropriate directories and create variables for both positive and negative, training/testing bag_of_words:
bag_pos_train, bag_pos_test = bag_of_words(positive)
bag_neg_train, bag_neg_test = bag_of_words(negative)



#VECTORIZE THE BAG_OG_WORDS_ARRAYS:

#create Vectorizers to vectorise the arrays of bag_of_words, using the CountVectorizer imported from sklearn:
train_vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 10000) 
test_vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 10000) 
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of strings.
total_training_Vectorized = train_vectorizer.fit_transform(bag_pos_train + bag_neg_train).toarray()

#Assign the vocabulary of the test vectorizer to be the same as the vocabulary of the train vectorizer:
test_vectorizer.vocabulary = train_vectorizer.get_feature_names()

total_testing_Vectorized = test_vectorizer.fit_transform(bag_pos_test + bag_neg_test).toarray()


#Create vectors with signs (1 = positive, -1=negative) for reviews, which will be combined with the vectorized arrays:
train_signs = np.empty(shape = (1600, 1))
train_signs[0:800,0] = 1
train_signs[800:1600, 0] = -1

test_signs = np.empty(shape = (400, 1))
test_signs[0:200,0] = 1
test_signs[200:400, 0] = -1

#Combine sign vectors with the vectorized bag_of_words arrays (signs will be the last column of the matrix):
combined_trainData_with_signs = np.concatenate((total_training_Vectorized, train_signs), axis = 1)
combined_testData_with_signs = np.concatenate((total_testing_Vectorized, test_signs), axis = 1)

#Call the train function (using the combined training array with signs) and create variables for positive and negative weights:
weights_positive, weights_negative = train(combined_trainData_with_signs)

#Call the testing function (using the combined testing array with signs):
total_accuracy = testing(combined_testData_with_signs, weights_positive, weights_negative)
#print(weights_positive)
print("\nAccuracy: {}".format(total_accuracy))

#TO GET THE MOST POSITIVELY WEIGHTED FEATURES:
#First, get feature names from the vectorizer:
feature_names_train = train_vectorizer.get_feature_names()
feature_names_test = test_vectorizer.get_feature_names()
#Find the index of the 10 highest values from the weight vectors (that were created using the train function):
sorted_weights_positive = sorted(range(len(weights_positive)), key=lambda i: weights_positive[i])[-10:]
sorted_weights_negative = sorted(range(len(weights_negative)), key=lambda i: weights_negative[i])[-10:]
#Use the index values found to get the words of those positions from the feature_names vectors:
print("\n10 most positively-weighted features for the positive class:")
for i in sorted_weights_positive:
    print("{}, {}". format(weights_positive[i], feature_names_train[i]))

print("\n10 most positively-weighted features for the negative class:")
for i in sorted_weights_negative:
    print("{}, {}". format(weights_negative[i], feature_names_test[i]))

    