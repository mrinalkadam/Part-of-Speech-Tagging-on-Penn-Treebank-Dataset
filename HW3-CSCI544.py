#!/usr/bin/env python
# coding: utf-8

# import required libraries and methods from them

from platform import python_version

import csv

import pandas as pd
import numpy as np

import re
import json
from collections import defaultdict

# check the python version being used by the jupyter notebook

python_version()

# read input file into a dataframe

START_TAG = '<s>'

def store_file_in_dataframe(file_name):
    sentences = []
    
    data_csv = csv.reader(open(file_name, 'r'), delimiter="\t", quoting=csv.QUOTE_NONE)
    
    row_original_words = [START_TAG]
    row_tags = [START_TAG]
        
    for row in data_csv:
        if len(row) == 0:
            sentences.append({'word_type': np.array(row_original_words),'POS_tag':np.array(row_tags)})
            
            row_original_words = [START_TAG]
            row_tags = [START_TAG]
            
            continue

        row_original_words.append(row[1])
        
        if len(row) == 2:
            row_tags = None
            
        else:
            row_tags.append(row[2])           
            
    if len(row) > 1:
        sentences.append({'word_type': np.array(row_original_words),'POS_tag':np.array(row_tags)})
      
    df = pd.DataFrame(columns=['word_type','POS_tag'],data=sentences)

    return df

# read training file into a dataframe

df_train = store_file_in_dataframe("data/train")


# # Task 1: Vocabulary Creation

# function to check for words that are entirely numbers and replace them with '<num>' tag using regex

def check_num_with_regex(words):
    num_reg = re.compile(r"[+-]?\d*\.?\d+")
    
    return np.array(list(map(lambda x:'<num>' if num_reg.fullmatch(x) is not None else x,words)))

# replace with <num> for df_train

df_train['mod_word_type'] = df_train['word_type'].apply(check_num_with_regex)

# create a dictionary with word:word_count as key:value pair

word_count_dict = {}

for index,row in df_train.iterrows():
    for word in row['mod_word_type']:
        word_count_dict[word] = word_count_dict.get(word,0) + 1

# # Selected threshold for unknown words replacement is 2 i.e. any word with a frequency of lesser than 2 will be considered as an unknown word.

# create a set of unknown words(words with frequency <= 1)

unknown_words = set()

for key in word_count_dict:
    if word_count_dict[key] <= 1:
        unknown_words.add(key)

# function to check for words that are unknown and replace them with '<unk>' tag in df_train

def replace_with_unk_train(df):
    for index,row in df_train.iterrows():
        temp = []
        for word in row['mod_word_type']:
            if word in unknown_words:
                temp.append('<unk>')
            else:
                temp.append(word)
        row['mod_word_type'] = temp

    return df_train

# replace with <unk> for df_train

df_train = replace_with_unk_train(df_train)

# calculate the sum of all <unk> and <num> tag words

new_word_count_dict = {}

unk_count = 0
num_count = 0

for key in word_count_dict.keys():
    # <num>
    if key == '<num>':
        num_val = word_count_dict[key]
        num_count += num_val  
    
    # <unk>
    if word_count_dict[key] <= 1:
        unk_val = word_count_dict[key]
        unk_count += unk_val
        
    else:
        new_word_count_dict[key] = word_count_dict[key]

word_count_sorted = [('<unk',unk_count),('<num>',num_count)] + list(sorted(new_word_count_dict.items(), key=lambda item: item[1],reverse=True))

# write out 'vocab.txt' file

with open("vocab.txt",'w') as f:
    csv_writer = csv.writer(f,delimiter='\t',quoting=csv.QUOTE_NONE)

    for i,(word,count) in enumerate(word_count_sorted):
        csv_writer.writerow((word,i,count)) 

# include <unk> and <num> tag words in the training vocab dictionary

new_word_count_dict.update({'<unk>':unk_count})
new_word_count_dict.update({'<num>':num_count})

print("Total size of my vocabulary:",len(new_word_count_dict))

# # Total size of my vocabulary : 21722

print("Total occurrences of the special token '< unk >' after replacement:",new_word_count_dict['<unk>'])

# # Total occurrences of the special token '< unk >' after replacement : 17952

print("Total occurrences of the special token '< num >' after replacement:",new_word_count_dict['<num>'])

# # Total occurrences of the special token '< num >' after replacement : 21241


# # Task 2: Model Learning

# calculate emission and transition(including initial) probabilities

emission_dd = defaultdict(int)
transition_dd = defaultdict(int)
tags_dd = defaultdict(int)

for index,row in df_train.iterrows():  
    
    # emission
    for i in range(1, len(row['POS_tag'])):
        emission_dd[row['POS_tag'][i],row['mod_word_type'][i]] += 1

    # transition
    for i in range(1, len(row['POS_tag'])):
        transition_dd[row['POS_tag'][i - 1], row['POS_tag'][i]] += 1

    # tags
    for tag in row['POS_tag']:
        tags_dd[tag] += 1

emission = defaultdict(float)
transition = defaultdict(float)
tags = set(tags_dd.keys())

for tag_1 in tags:
    for tag_2 in tags:
        transition_dd[tag_1, tag_2]

for key,count in emission_dd.items():
    emission[key] = count / tags_dd[key[0]]

# apply smoothing for transition probabilities

for key,count in transition_dd.items():
    transition[key] = (count + 1) / (tags_dd[key[0]] + len(tags))

# length of transition dictionary(transition dictionary includes initital transition probabilities from start tag as well)
# start tag <s> has also been included and smoothing has been applied - hence a 46*46 dictionary

print(len(transition))

# # Transition parameters in my HMM : 2116

# length of emission dictionary

print(len(emission))

# # Emission parameters in my HMM : 28831

# format the keys of both dictionaries

emission_new = dict((key[0] + ',' + key[1], value) for (key, value) in emission.items())
transition_new = dict((key[0] + ',' + key[1], value) for (key, value) in transition.items())

# write out 'hmm.json' file

with open('hmm.json', 'w') as f:
        json.dump({
            'transition': dict((key,val) for key, val in transition_new.items()),
            'emission': dict((key,val) for key, val in emission_new.items())
        }, f, indent=2)


# # Task 3: Greedy Decoding with HMM

# read dev(validation) file into a dataframe

df_dev = store_file_in_dataframe("data/dev")

# replace with <num> for df_dev

df_dev['mod_word_type'] = df_dev['word_type'].apply(check_num_with_regex)

# function to check for words that are unknown and replace them with '<unk>' tag

def replace_with_unk(df):
    for index,row in df.iterrows():
        new_words = []
        for word in row['mod_word_type']:
            if word not in new_word_count_dict.keys():
                word = '<unk>'
            new_words.append(word)

        row['mod_word_type'] = new_words

    return df

# replace with <unk> for df_dev

df_dev = replace_with_unk(df_dev)

# function to return predicted POS tags using greedy decoding algorithm

def greedy_decoding(df):
    results = []
    
    for index,row in df.iterrows():
        output_states = [START_TAG]
        for i in range(1,len(row['mod_word_type'])):
            max_seen = -1
            max_tag = ''
            
            for tag in tags:
                temp = transition[output_states[-1], tag] * emission[tag, row['mod_word_type'][i]]
                
                if temp > max_seen:
                    max_seen = temp
                    max_tag = tag
            
            output_states.append(max_tag)
        results.append(output_states)
    
    return results

# function to store the predicted POS tags into a new column of the dataframe

def store_predicted_POS_tags_as_new_column_in_df(algorithm,df):
    predicted_tags = algorithm(df)

    df['predicted_POS_tag'] = ''
    for i in range(len(df)):
        df['predicted_POS_tag'].iloc[i] = predicted_tags[i]

    return df

# store the predicted POS tags using greedy algorithm into a new column in df_dev

df_dev = store_predicted_POS_tags_as_new_column_in_df(greedy_decoding,df_dev)

# function to check accuracy of predicted tags with true tags

def accuracy(df):
    count_correct = 0
    count_incorrect = 0
    
    for index,row in df.iterrows():
        for tag_index in range(1,len(row['POS_tag'])):
            if row['POS_tag'][tag_index] == row['predicted_POS_tag'][tag_index]:
                count_correct += 1
            else:
                count_incorrect += 1
        
    acc = np.round((count_correct/(count_correct+count_incorrect))*100,2)
    
    return acc

# find out the accuracy of greedy algorithm on dev data

acc = accuracy(df_dev)
print("Accuracy of greedy algorithm on dev:",acc,"%")

# # Accuracy of greedy algorithm on dev data : 93.69 %

# read test file into a dataframe

df_test = store_file_in_dataframe("data/test")

# replace with <num> for df_test

df_test['mod_word_type'] = df_test['word_type'].apply(check_num_with_regex)

# replace with <unk> for df_test

df_test = replace_with_unk(df_test)

# store the predicted POS tags using greedy algorithm into a new column in df_test

df_test = store_predicted_POS_tags_as_new_column_in_df(greedy_decoding,df_test)

# write out 'greedy.out' file

with open("greedy.out",'w') as f:
    csv_writer = csv.writer(f,delimiter='\t',quoting=csv.QUOTE_NONE)

    for index,row in df_test.iterrows():
        for i in range(1,len(row['word_type'])):
            csv_writer.writerow((i,row['word_type'][i],row['predicted_POS_tag'][i]))
        csv_writer.writerow([])


# # Task 4: Viterbi Decoding with HMM

# function to return predicted POS tags using Viterbi decoding algorithm

def viterbi_decoding(df): 
    results = []
    
    for index,row in df.iterrows():
        vit = defaultdict(lambda: defaultdict(lambda: (0.0, '')))

        for tag in tags:
            vit[1][tag] = (transition[START_TAG, tag] * emission[tag, row['mod_word_type'][1]], START_TAG)

        for j in range(2, len(row['mod_word_type'])):
            word = row['mod_word_type'][j]
            
            for tag in tags:
                max_seen = -1
                max_tag = ''
                
                for last_tag in vit[j - 1].keys():
                    temp = vit[j - 1][last_tag][0] * transition[last_tag, tag] * emission[tag, word]
                
                    if temp > max_seen:
                        max_seen = temp
                        max_tag = last_tag
                        
                vit[j][tag] = (max_seen, max_tag)  

        # backtracking
        
        output_states = []
        
        j = len(row['mod_word_type']) - 1
        tmp = max(vit[j].items(), key=lambda x: x[1][0])
        tag = tmp[0]
        last_value = tmp[1]
        
        output_states.append(tag)
        
        j -= 1

        while j >= 0:
            tag = last_value[1]
            last_value = vit[j][tag]
            
            output_states.append(tag)
            
            j -= 1

        output_states.reverse()
    
        results.append(output_states)
        
    return results

# drop previously predicted POS tags by greedy algorithm from df_dev

df_dev.drop(['predicted_POS_tag'],axis=1,inplace = True)

# store the predicted POS tags using Viterbi algorithm into a new column in df_dev

df_dev = store_predicted_POS_tags_as_new_column_in_df(viterbi_decoding,df_dev)

# find out the accuracy of Viterbi algorithm on dev data

acc = accuracy(df_dev)
print("Accuracy of Viterbi algorithm on dev:",acc,"%")

# # Accuracy of Viterbi algorithm on dev data : 94.95 %

# drop previously predicted POS tags by greedy algorithm from df_test

df_test.drop(['predicted_POS_tag'],axis=1,inplace = True)

# store the predicted POS tags using Viterbi algorithm into a new column in df_test

df_test = store_predicted_POS_tags_as_new_column_in_df(viterbi_decoding,df_test)

# write out 'viterbi.out' file

with open("viterbi.out",'w') as f:
    csv_writer = csv.writer(f,delimiter='\t',quoting=csv.QUOTE_NONE)

    for index,row in df_test.iterrows():
        for i in range(1,len(row['word_type'])):
            csv_writer.writerow((i,row['word_type'][i],row['predicted_POS_tag'][i]))
        csv_writer.writerow([])



