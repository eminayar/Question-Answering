import os
import sys
import tfidf_model as tf_idf
import stanfordnlp as sn
import pickle
import numpy as np
from gensim.models import KeyedVectors

lang = 'tr'
treebank_shorthand = 'tr_imst'
punctuations = {'#', '[', '~', '-', ']', '.', '@', '/', "'", '{', '|', ')',
                '(', '*', ',', '`', ';', '$', '%', '\\', '^', '_', '!', '<', ':', '&', '>', '"', '}', '=', '?', '+'}
question_words = {"hangi", "nedir", "ne", "kadar", "neden", "nasıl", "nerede", "nerelerde", "neyi",
                  "kadarı", "nereye", "ne zaman", "neyle", "kaç", "hangisidir", "mıdır", "kaça",
                  "neresidir", "nelerdir", "nerelerdir", "nereden", "nelerdir", "mı", "mi", "kaçtır",
                  "kim", "kaçını", "kaçıncı", "adlandırılır", "neyin", "kimin", "kaçtı",
                  "kimler", "kimler tarafından", "nerelere", "neye", "hangileridir", "nereyi", "hangisinin",
                  "nelere", "neresinde", "kimdir", "nelerden", "neyden", "nelerin"}

stopwords = {"bu", "şu", "ve", "bir", "denir", "nedenle", "olarak", "ilk", "de",
             "da", "dan", "den", "ama", "ancak", "eğer", "böyle", "epey", "dolayı",
             "öteki", "illa", "illaki", "kısaca", "öyle", "öylece", "örneğin", "görsel",
             "buna", "karşın", "grafik", "diğer", "daha", "ise", "için", "ile", "resim",
             "ayrıca", "iken", "kadar", "fotoğraf", "şimdi", "en", "budur", "bağlı", "sonucunda"}

VECTOR_LEN = 400
GROUP_ID = sys.argv[0].split("/")[-1].split(".")[0]

PATH = '/'.join(sys.argv[0].split("/")[:-1])
models_directory = PATH + '/stanfordnlp_resources'
table_file = open(PATH+'/table_file', 'rb')
idf_file = open(PATH+'/idf_file', 'rb')
word_vectors = KeyedVectors.load_word2vec_format(PATH+'/trmodel', binary=True)
table = pickle.load(table_file)
idf = pickle.load(idf_file)
nlp = sn.Pipeline(lang=lang, treebank=treebank_shorthand,
                  models_dir=models_directory, processors='tokenize,lemma')
paragraphs = pickle.load(open(PATH+'/paragraphs_file', 'rb'))


def tokenize(input):
    input = nlp(input)
    tokens = []
    for sentence in range(len(input.sentences)):
        for token in range(len(input.sentences[sentence]._tokens)):
            if input.sentences[sentence]._tokens[token].words[0]._text not in punctuations:
                tokens.append(
                    input.sentences[sentence]._tokens[token].words[0].lemma)
    return tokens


QUESTION_PATH = sys.argv[1]
TASK1 = open(sys.argv[2]+'/'+GROUP_ID+'.txt' , 'w', encoding='utf16')
TASK2 = open(sys.argv[3]+'/'+GROUP_ID+'.txt', 'w', encoding='utf16')

print("ANSWERING QUESTIONS")
num_lines = sum(1 for line in open(QUESTION_PATH, encoding='utf16'))
counter = 0
for line in open(QUESTION_PATH, encoding='utf16'):
    counter += 1
    print(str(100*counter/num_lines), end='\r')
    query = line
    query_nlp = nlp(query)
    query = tf_idf.computeTF_IDF(tf_idf.computeTF(tokenize(query)), idf)
    query_representation = np.zeros(VECTOR_LEN)
    word_count = 0
    for token in query_nlp.sentences[0]._tokens:
        word = token.words[0].lemma
        if word not in word_vectors:
            continue
        if word in question_words or token.words[0] in question_words:
            continue
        word_count += 1
        query_representation += word_vectors[word]
    if word_count > 0:
        query_representation /= word_count
    answers = []
    for paragraph in table:
        similarity = tf_idf.cos_similarity(query, table[paragraph])
        answers.append((similarity, paragraph))
    answers = sorted(answers, reverse=True)[:5]
    sentence_scores = []
    for answer in answers:
        current_paragraph = paragraphs[answer[1]]
        similarity_score = answer[0]
        for sentence in current_paragraph:
            sentence_representation = np.zeros(VECTOR_LEN)
            word_count = 0
            for token in sentence._tokens:
                word = token.words[0].lemma
                if word not in word_vectors:
                    continue
                word_count += 1
                sentence_representation += word_vectors[word]
            if word_count > 0:
                sentence_representation /= word_count
            denom = (np.linalg.norm(sentence_representation)
                     * np.linalg.norm(query_representation))
            score = np.dot(query_representation, sentence_representation)
            if denom > 0:
                score /= denom
            sentence_scores.append(
                (score + 0.75 * similarity_score, answer[1], sentence))
    ans = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[0]
    TASK1.write(str(ans[1])+'\n')
    sentence = ans[2]
    N = 0
    mapper = {}
    cnt = 0
    for word in sentence._tokens:
        if word.words[0]._text not in punctuations and str(word.words[0].lemma).lower() not in stopwords and str(word.words[0].lemma).lower() not in question_words:
            mapper[N] = cnt
            N += 1
        cnt += 1
    interval = (0, 0)
    best = 0
    for i in range(N):
        for j in range(N):
            if i > j:
                continue
            sentence_rep = np.zeros(VECTOR_LEN)
            word_count = 0
            for k in range(N):
                word = sentence._tokens[mapper[k]].words[0].lemma
                if k >= i and k <= j:
                    continue
                if word not in word_vectors:
                    continue
                sentence_rep += word_vectors[word]
                word_count += 1
            if word_count > 0:
                sentence_rep /= word_count
            denom = (np.linalg.norm(sentence_rep) *
                     np.linalg.norm(query_representation))
            score = np.dot(query_representation, sentence_rep)
            if denom > 0:
                score /= denom
            if score > best:
                best = score
                interval = (mapper[i], mapper[j])

    t2_ans = ""
    for i in range(interval[0], interval[1]+1):
        t2_ans += sentence.words[i]._text + " "
    t2_ans = t2_ans.strip()
    if len(t2_ans) < 2:
        t2_ans = "FAIL"
    TASK2.write(t2_ans + '\n')
