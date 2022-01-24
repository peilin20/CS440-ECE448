# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
from re import M
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.78,silently=False):
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)
    #training phase
    pos_dict, neg_dict, pos_totalnum, neg_totalnum = construct_dict(train_set, train_labels)
    
    pos_log_prob_dict = {}
    neg_log_prob_dict = {}

    #calculate prob for positive words
    pos_prob=laplace / (pos_totalnum + laplace*(len(pos_dict)+1))
    log_pos_prob=math.log(pos_prob)
    for words in pos_dict:
        pos_log_prob = math.log(pos_dict[words]*pos_prob/laplace + pos_prob)
        pos_log_prob_dict[words] = pos_log_prob
    #calculate prob for negative words
    neg_prob=laplace / (neg_totalnum + laplace*(len(neg_dict)+1))
    log_neg_prob=math.log(neg_prob)
    for words in neg_dict:
        neg_log_prob = math.log(neg_dict[words]*neg_prob/laplace + neg_prob)
        neg_log_prob_dict[words] = neg_log_prob

    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        pos=math.log(pos_prior)
        neg=math.log(1-pos_prior)
        for words in doc:
            if words in pos_dict:
                pos+=pos_log_prob_dict[words]
            else:
                pos+=log_pos_prob
            if words in neg_dict:
                neg+=neg_log_prob_dict[words]
            else:
                neg+=log_neg_prob
        yhats.append((neg<pos))
    return yhats

def construct_dict(training_set,training_label):
    pos_dict = {}
    neg_dict = {}
    pos_count=0
    neg_count=0
    i=0
    for label in training_label:
        review=training_set[i]
        
        for words in review:
            if training_label[i]==1:
                pos_count+=1
                if words in pos_dict:
                    pos_dict[words] +=1
                else:
                    pos_dict[words] =1
            else:
                neg_count+=1
                if words in neg_dict:
                    neg_dict[words] +=1
                else:
                    neg_dict[words]=1
        i+=1
    return pos_dict, neg_dict, pos_count,neg_count
        
        
# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.1, bigram_laplace=0.001, bigram_lambda=0.3,pos_prior=0.5, silently=False):
    #construct dictionary for bigram
    bigpos_dict={}
    bigneg_dict={}
    bigpos_totalnum, bigneg_totalnum=0,0
    for i in range(len(train_labels)):
        review = train_set[i]
        for j in range(len(review)-1):
            if train_labels[i]==1:
                bigpos_totalnum+=1
                if(review[j],review[j+1]) in bigpos_dict:
                    bigpos_dict[(review[j],review[j+1])] +=1
                else:
                    bigpos_dict[(review[j],review[j+1])] =1
            else:
                bigneg_totalnum+=1
                if(review[j],review[j+1]) in bigneg_dict:
                    bigneg_dict[(review[j],review[j+1])] +=1
                else:
                    bigneg_dict[(review[j],review[j+1])] =1

    
    #calculate prob for positive words
    bigpos_log_prob_dict, biglog_pos_prob = construct_prob_dict(bigpos_dict,bigram_laplace,bigpos_totalnum)
    #calculate prob for negative words
    bigneg_log_prob_dict, biglog_neg_prob = construct_prob_dict(bigneg_dict,bigram_laplace,bigneg_totalnum)

    #construct unigram
    pos_dict, neg_dict, pos_totalnum, neg_totalnum = construct_dict(train_set, train_labels)
    unipos_log_prob_dict, log_pos_prob = construct_prob_dict(pos_dict,unigram_laplace,pos_totalnum)
    unineg_log_prob_dict, log_neg_prob = construct_prob_dict(neg_dict,unigram_laplace,neg_totalnum)

    # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        #first start with unigram
        uni_pos=math.log(pos_prior)
        uni_neg=math.log(1-pos_prior)
        for words in doc:
            if words in pos_dict:
                uni_pos+=unipos_log_prob_dict[words]
            else:
                uni_pos+=log_pos_prob
            if words in neg_dict:
                uni_neg+=unineg_log_prob_dict[words]
            else:
                uni_neg+=log_neg_prob
        #then bigram
        big_pos=math.log(pos_prior)
        big_neg=math.log(1-pos_prior)
        for i in range(len(doc)-1):
            if (doc[i],doc[i+1]) in bigpos_dict:
                big_pos += bigpos_log_prob_dict[(doc[i],doc[i+1])]
            else:
                big_pos += biglog_pos_prob
            if (doc[i],doc[i+1]) in bigneg_dict:
                big_neg += bigneg_log_prob_dict[(doc[i],doc[i+1])]
            else:
                big_neg += biglog_neg_prob
        
        p = bigram_lambda* big_pos + uni_pos * (1-bigram_lambda)
        n = bigram_lambda* big_neg + uni_neg * (1-bigram_lambda)


        yhats.append((n<p))
    return yhats

def construct_prob_dict(dict,laplace,word_count):
    log_prob_dict = {}
    
    unknown_p = laplace/(word_count + laplace*(len(dict) +1))
    for words in dict:
        log_prob = dict[words]*unknown_p/laplace + unknown_p
        log_prob_dict[words] = math.log(log_prob)
    
    return log_prob_dict, math.log(unknown_p)