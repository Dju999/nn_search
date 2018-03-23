# coding:utf-8
import os
import logging
import pickle

import numpy as np
import pymorphy2
import string
morph = pymorphy2.MorphAnalyzer()

from gensim.models import KeyedVectors, Word2Vec
import nltk
from nltk import word_tokenize

nltk.download('stopwords')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

f_name = 'wv_keyed_vectors.w2v'
corpus_file = 'ivi_corpus.pkl'
d2v_file = 'doc2vec.pkl'
content_descr_file = 'content_descr.tsv'

if os.path.isfile(f_name):
    word_vectors = KeyedVectors.load(f_name)
else:
    if os.path.isfile(corpus_file):
        text_corpus = pickle.load(open(corpus_file, 'rb'))
    else:
        txt = open(content_descr_file, 'r').readlines()
        corpus = dict()
        n = 0
        for t in txt:
            if n % 1000 == 0:
                logger.info("doc {}".format(n))
            n += 1
            words = np.array(word_tokenize(t))
            corpus.update({words[0]: [
                morph.parse(str(i).lower())[0].normal_form for i in words[1:]
                if (i not in list(string.punctuation)+['«','»','–']
                    and i not in nltk.corpus.stopwords.words('russian'))
            ]})
        pickle.dump(corpus, open(corpus_file, 'wb'), protocol=2)
    model = Word2Vec(sentences=corpus.values(), size=100, window=5, min_count=5, workers=10)
    word_vectors = model.wv
    del model
    word_vectors.save(f_name)
print("любовь {}".format(word_vectors.most_similar('любовь')))
logger.info("Векторизуем корпус текстов {}")
if os.path.isfile(d2v_file):
    doc2vec = pickle.load(open(d2v_file, 'rb'))
else:
    prepared_text_corpus = pickle.load(open(corpus_file, 'rb'))
    doc2vec = dict()
    for k in prepared_text_corpus:
        raw_tokens = prepared_text_corpus[k]
        token_vectors = np.array([word_vectors[t] for t in raw_tokens])
        d2v = token_vectors.mean(axis=0)
        doc2vec[k] = d2v
    pickle.dump(doc2vec, open(d2v_file, 'wb'), protocol=2)
print(doc2vec[list(doc2vec.keys())[1]])