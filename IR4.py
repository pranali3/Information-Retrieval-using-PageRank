
# coding: utf-8

# In[11]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from functools import lru_cache
import os
import numpy as np
import pandas as pd
import math
import string
from nltk.util import ngrams
import re



#working
def tokenize(path):
    token_dict = {}
    ngrams_dict = {}
    tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ','.']
    stemmer  = PorterStemmer()
    stem = lru_cache(maxsize=50000)(stemmer.stem)
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if(os.path.isfile(file_path)):
            with open(file_path, "r") as file:
                tokens = []
                tagged_tokens = []
                for line in file:
                    tokens.extend(line.split(" "))
                tagged_tokens = [word.strip().split("_") for word in tokens if word.strip()!=""]
                #ngrams_dict[filename] = []
                preprocessed_tokens = []
                #remove punctuations except '-' and '.' and lower()
                preprocessed_tokens = [ re.sub('[^\w\s\-\.]','',x[0].lower()) for x in tagged_tokens]
                #stemming
                preprocessed_tokens = [ stem(x) for x in preprocessed_tokens]
                #form ngrams
                ngrams_dict[filename] = []
                ngrams_dict[filename].append(preprocessed_tokens) #unigrams with stopwords and all tags
                ngrams_dict[filename].append(list(ngrams(preprocessed_tokens, 2))) #bigrams with stopwords and all tags
                ngrams_dict[filename].append(list(ngrams(preprocessed_tokens, 3))) #trigrams with stopwords and all tags
                tokens = [re.sub('[^\w\s\-\.]','',x[0].lower()) for x in tagged_tokens if x[1] in tags] 
                tokens = [token for token in tokens if token not in (stopwords.words('english'))] #without stopwords               
                tokens = [stem(token).strip(" -_") for token in tokens]
                tokens = [token for token in tokens if token not in (stopwords.words('english'))]
                token_dict[filename] = tokens
                
                
    return token_dict, ngrams_dict       
                    
    

#working
path = 'C:\\Users\\prana\\www\\gold'
def preprocess_gold(path):
    gold_dict = {}
    stemmer  = PorterStemmer()
    stem = lru_cache(maxsize=50000)(stemmer.stem)
    for filename in os.listdir(path):
        if not filename.startswith('.'):
            file_path = os.path.join(path,filename)
            if (os.path.isfile(file_path)):
                with open(file_path, "r") as file:
                    gold_dict[filename] = []
                    for line in file:
                        line = [ re.sub('[^\w\s\-\.]','',x) for x in line.split()]
                        gold_dict[filename].append(" ".join([stem(x) for x in line]))
                    
    return gold_dict



def create_adj_map(tokens, window, adj_map):
    
    if len(tokens) == 1:
        adj_map = {next(iter(tokens)):None}
        return adj_map
    
        
    for item in window:
        if item in tokens:
            if (window.index(item))+1 != len(window) and (tokens.index(item))+1 != len(tokens):
                next_word = window[window.index(item)+1]
                if item in adj_map:
                    if next_word in adj_map[item]:
                        adj_map[item][next_word] += 1
                    elif next_word == tokens[tokens.index(item)+1]:
                        adj_map[item][next_word] = 1
                else:
                    adj_map[item] = {}
                    if next_word == tokens[tokens.index(item)+1]:
                        adj_map[item][next_word] = 1
                
                #since undirected graph
                if next_word in adj_map:
                    if item in adj_map[next_word]:
                        adj_map[next_word][item] += 1
                    elif next_word == tokens[tokens.index(item)+1]:
                        adj_map[next_word][item] = 1
                else:
                    adj_map[next_word] = {}
                    if next_word == tokens[tokens.index(item)+1]:
                        adj_map[next_word][item] = 1
    
    adj_map.pop('.', None)       
    for key in adj_map:
        adj_map[key].pop('.', None)
            

def PageRank(scores, alpha, adj_map, iteration):
    iteration += 1
    n = len(scores)
    p = 1/n
    
    for key in scores:
        total = 0.0
        if key in adj_map and adj_map[key]:
            for value in adj_map[key]:
                num = (adj_map[value][key] * scores[value])
                deno = 0
                for val_of_val in adj_map[value]: #val_of_val is k, value is j, key is i
                    deno += adj_map[value][val_of_val]
                total += (num/deno)
        scores[key] = alpha * total + (1-alpha)*p
    if iteration == 10:
        return
    else:
        PageRank(scores, alpha, adj_map, iteration)
    return scores  


def add_ngrams(scores, ngrams):
    for item in ngrams[1]:
        if(item[0] in scores and item[1] in scores):#ngrams with stopwords are skipped since absent in scores
            scores[item] = scores[item[0]] + scores[item[1]]
    for item in ngrams[2]:
        if(item[0] in scores and item[1] in scores and item[2] in scores):
            scores[item] = scores[item[0]] + scores[item[1]] + scores[item[2]]
            

def predict_keyphrases(token_dict, gold_dict, ngrams_dict, alpha, w):
    pred_list = {}
    window = []
    
    for doc in token_dict:
        if doc in gold_dict:
            newList = ngrams_dict[doc][0]
            adj_map = {} #new map for every doc
            for word in ngrams_dict[doc][0]:
                window = [] #new window starting from every word in doc
                for i in range(int(w)):
                    if (newList.index(word)+i) == len(newList):
                        break
                    else:
                        item = newList[newList.index(word)+i]
                        window.append(item)
                create_adj_map(token_dict[doc], window,adj_map)
                           
        nodes = set(token_dict[doc])
        if '.' in nodes:
            nodes.remove('.')
            n = len(nodes)
            init_scores = {}
            for word in token_dict[doc]:
                if word!= '.' and word not in init_scores:
                    init_scores[word] = 1/n
            pr_scores = PageRank(init_scores, alpha, adj_map, 0)
            add_ngrams(pr_scores, ngrams_dict[doc]) #only ngrams without stopwords are considered
            sorted_values = sorted(pr_scores, key = pr_scores.get , reverse = True)
            # gold phrases with stopwords are skipped (not matched)
            pred_list[doc] = [" ".join(x) if not isinstance(x, str) else x for x in sorted_values]
            
    return pred_list
    

def calc_doc_MRR(gold_list, pred_list):
    for i in range(len(pred_list)):
        if pred_list[i] in gold_list:
            return 1/(i+1)
    return 0   

def calc_global_MRR(gold_dict, pred_list, k):
    MRR_values = np.zeros(k)
    for doc in pred_list:
        for i in range(k):
            MRR_values[i] += calc_doc_MRR(gold_dict[doc], pred_list[doc][:i+1])
    return MRR_values/len(gold_dict) 


if __name__ == "__main__":
    
    print("########################################")
    print("CS 582: Assignment 4 ")
    print("Name: Pranali Loke        NetId: ploke2")
    print("########################################")
    print("\n")
	
	abstracts_path = ""
    gold_path = ""
    
    while True:
        abstracts_path = input("Enter path to the directory where abstracts are stored:\n")
        if(os.path.isdir(abstracts_path)):
            break
        else:
            print("Invalid path\n")
            
    while True:
        gold_path = input("Enter path to the directory where gold standard documents are stored:\n")
        if(os.path.isdir(gold_path)):
            break
        else:
            print("Invalid path\n")
            
    
    token_dict, ngrams_dict = tokenize(abstracts_path)
    gold_dict = preprocess_gold(gold_path)
    
    absent_gold_docs = []
    present_gold_docs = []
    
    for key in token_dict:
        if key not in gold_dict:
            absent_gold_docs.append(key)
        else:
            present_gold_docs.append(key)
            
    for key in absent_gold_docs:
        token_dict.pop(key, None)
        ngrams_dict.pop(key, None)
        
    alpha = 0.85
    k = 10
    
    w = input("Enter the window size:\n")
    
    pred_list = predict_keyphrases(token_dict, gold_dict, ngrams_dict, alpha, w)
    MRR_values = calc_global_MRR(gold_dict, pred_list, k)
    
    print("\nDisplaying MRR values for k = 1 to 10...")
    for i in range(len(MRR_values)):
        print("k = " + str(i+1) + "\tMRR = " + str(MRR_values[i])) 
    

