import os
import tfidf_model as tf_idf
import numpy as np
import stanfordnlp as sn
import pickle

###path to dataset
DATASET_PATH = 'path/to/derlem.txt'
###path to standford models
models_directory = 'stanfordnlp_resources'
lang = 'tr'
treebank_shorthand = 'tr_imst'
paragraphs = {}
punctuations = {'#', '[', '~', '-', ']', '.', '@', '/', "'", '{', '|', ')',
                '(', '*', ',', '`', ';', '$', '%', '\\', '^', '_', '!', '<', ':', '&', '>', '"', '}', '=', '?', '+'}

def tokenize(parag_num, input):
    input = input.lower()
    input = nlp(input)
    paragraphs[parag_num] = input.sentences
    tokens = []
    for sentence in range(len(input.sentences)):
        for token in range(len(input.sentences[sentence]._tokens)):
            if input.sentences[sentence]._tokens[token].words[0]._text not in punctuations:
                tokens.append(input.sentences[sentence]._tokens[token].words[0].lemma)
    return tokens

nlp = sn.Pipeline( lang=lang, treebank=treebank_shorthand, models_dir=models_directory , processors='tokenize,lemma')
dataset = open(DATASET_PATH, encoding='utf16')
paragraphs_by_tokens = []
counter = 0 

for line in dataset:
    counter += 1
    print('%'+str(100*counter/7612), end='\r')
    line = line.strip()
    if line == '':
        continue
    line = line.split(" ", 1)
    parag_num = int(line[0])
    line = line[1]
    if line[-1] not in punctuations:
        continue
    tokens = tokenize( parag_num, line )
    paragraphs_by_tokens.append({'id': parag_num , 'tokens': tokens})

table, idf = tf_idf.tf_idf_model(paragraphs_by_tokens)
table_file = open('table_file', 'wb')
idf_file = open('idf_file', 'wb')
paragraphs_file = open('paragraphs_file', 'wb')
pickle.dump(table, table_file)
pickle.dump(idf, idf_file)
pickle.dump(paragraphs, paragraphs_file)