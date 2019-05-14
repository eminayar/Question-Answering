import os
import lemmatizer as lm
import tfidf_model as tf_idf
import numpy as np

DATASET_PATH = 'deliverables/derlem.txt'
QUESTION_GROUPS_PATH = 'deliverables/soru_gruplari.txt'
punctuations = {'#', '[', '~', '-', ']', '.', '@', '/', "'", '{', '|', ')',
                '(', '*', ',', '`', ';', '$', '%', '\\', '^', '_', '!', '<', ':', '&', '>', '"', '}', '=', '?', '+'}

def lemmatize( input ):
    finding = lm.findPos(input.lower(), lm.revisedDict)[0][0]
    return finding.split('_')[0]

def tokenize(input):
    input = input.strip()
    without_punc = ""
    for char in input:
        if char in punctuations:
            without_punc += ' '
        else:
            without_punc += char
    tokens = list(map(lambda x: x.lower(), without_punc.split()))
    tokens = list(map(lambda x: lemmatize(x), tokens))
    return tokens

dataset = open(DATASET_PATH, encoding='utf16')
paragraphs_by_tokens = []
for line in dataset:
    line = line.strip()
    if line == '':
        continue
    tokens = tokenize( line )
    parag_num = int(tokens[0])
    tokens = tokens[1:]
    paragraphs_by_tokens.append({'id': parag_num , 'tokens': tokens})

table, idf = tf_idf.tf_idf_model(paragraphs_by_tokens)

query = "Güneş sistemindeki 5. büyük gezegenin adı nedir?"
query = tf_idf.computeTF_IDF( tf_idf.computeTF( tokenize( query ) ), idf )

# res = np.argmin(tf_idf.cos_similarity(query, table[paragraph]) for paragraph in table)
# closest = -1
# answer = -1
# for paragraph in table:
#     diff = tf_idf.cos_similarity(query, table[paragraph] )
#     if diff:
#         print(paragraph, diff)
#     if closest == -1 or diff > closest:
#         closest = diff
#         answer = paragraph

# print(answer)

answers = []
for paragraph in table:
    diff = tf_idf.cos_similarity(query, table[paragraph] )
    answers.append((paragraph,diff))

answers = sorted(answers, key=lambda x:x[1], reverse=True)[:5]
print(answers)