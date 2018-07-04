# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 12:34:01 2017

@author: hardy
"""
import math
import time
from collections import defaultdict
from itertools import product
from copy import copy
import nltk
import random

random.seed(8122017)
R = random.random()
tag_set = ['.', 'NUM', 'VERB', 'DET', 'ADV', 'CONJ', 'PRT', 'PRON', 'ADP', 'X', 'NOUN', 'ADJ']
corpus = nltk.corpus.brown.tagged_sents(tagset='universal')   
processed_corpus = [sent for sent in corpus if len(sent) < 5]

def train(tag_set, processed_corpus):
    """Training using Structure Perceptron"""
    data_pairs = list(extract_corpus('train', processed_corpus))
    W = defaultdict(int)
    W_Sum = defaultdict(int)
    sum_all = 0
    print('Training')
    for n in range(0, 10):
        print('Iteration: ' + str(n))
        random.shuffle(data_pairs, lambda: R)
        for i in range(0, len(data_pairs)):
            sent, tags = zip(*data_pairs[i])
            big_phi_z, z = arg_max(sent, W, tag_set)
            if not isSame(tags, z):
                big_phi_t = process_feature(tags, sent)
                update_parameters(W, big_phi_t, big_phi_z)
            for key, value in W.items():
                W_Sum[key] += value
            # print('Finished generating sentence ' +str(i)+ ' of ' + str(len(data_pairs)) + '\n') 
            sum_all +=1
    for key, value in W_Sum.items():
        W_Sum[key] = value / sum_all
    return W_Sum

def test(W, tag_set, processed_corpus):
    """Testing using Structure Prediction"""
    print('Testing')
    data_pairs = list(extract_corpus('test', processed_corpus))
    correct = 0
    sum_all = 0
    for i in range(0, len(data_pairs)):
        sent, tags = zip(*data_pairs[i])
        big_phi_z, z = arg_max(sent, W, tag_set)
        for j in range(0, len(tags)):
            sum_all += 1
            if tags[j] == z[j]:
                correct += 1
        #print('Finished generating sentence ' +str(i)+ ' of ' + str(len(data_pairs)) + '\n')
    print('Accuracy: ' + str(correct / sum_all * 100) + '%\n')

def isSame(t, z):
    for j in range(0, len(t)):
        if t[j] != z[j]:
            return False
    return True

def reorder_product(*a, rep):
    for tup in product(*a[::-1], repeat=rep):
        yield tup[::-1]

def arg_max(sent, W, tag_set):
    """Calculating the arg_max from all possible combination of tags and words"""
    tag_set = tag_set[::-1]
    n_comb = math.pow(len(tag_set), len(sent))
    max_result = float('-inf')
    save_big_phi = None
    save_tags = None
    tags_generator = reorder_product(tag_set, rep=len(sent))
    for j in range(0, int(n_comb)):
        result = 0
        tags = next(tags_generator)
        big_phi = process_feature(tags, sent)
        for feature, count in big_phi.items():
            if feature in W:
                result += W[feature] * count
        if result > max_result:
            max_result = result
            save_big_phi = big_phi
            save_tags = tags
    return save_big_phi, save_tags

def update_parameters(W, big_phi_t, big_phi_z):
    """Update W parameters"""
    for key, value in big_phi_t.items():
        W[key] += value
    for key, value in big_phi_z.items():
        W[key] -= value
    return W

def extract_corpus(mode, corpus):
    """Extract sentence which length < 5 from Brown's corpus """
    if mode == 'train':
        return corpus[:2500]
    else:
        return corpus[2500:]

def process_feature(tags, sent):
    """ Process sents into features"""
    big_phi = defaultdict(int)
    for i in range(len(sent)):
        if i == 0:
            big_phi[tags[i] + '<<START>>'] += 1
        else:
            big_phi[tags[i] + tags[i-1]] += 1
        big_phi[sent[i] + tags[i]] += 1
    return big_phi

timing = time.time()
W = train(tag_set, processed_corpus)
test(W, tag_set, processed_corpus)
print(str(time.time() - timing))
