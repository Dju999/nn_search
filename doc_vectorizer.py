# coding:utf-8

import pandas as pd
import numpy as np
import nltk
import pymorphy2
from gensim.models.word2vec import Word2Vec, FastText
import string
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

morph = pymorphy2.MorphAnalyzer()
# при ошибках импорта запустить nltk.download - 3 гига
# nltk.download()
if __name__ == '__main__':
    raw_text = pd.read_csv('./data/content_description.tsv', sep='\t', index_col='id')
    # tokenize = raw_text.head(2).apply(
    #     lambda x: list([morph.parse(i)[0].normal_form for i in nltk.word_tokenize(' '.join(x if isinstance(x, str) else "" ))]), axis=1).values
    # https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
    # print(tokenize)
    # print(raw_text.head(2).apply(lambda x: ' '.join(x if isinstance(x, str) else ""), axis=1))
    docs = (raw_text.apply(
        lambda x: [
            morph.parse(j)[0].normal_form for j in
            nltk.word_tokenize(' '.join([i for i in x if isinstance(i, str)]))
            if j not in list(string.punctuation)
        ],
        axis=1).apply(np.unique).values
    )
    # http://rusvectores.org/ru/models/
    w2v = FastText.load_fasttext_format(wiki.ru.vec)
    w2v.init_sims(replace=True)
    #print(dir(model))
    #print(model.vocab)
    for i in docs[1]:
        try:
            print(i)
            print(w2v.get_vector(i))
        except KeyError:
            pass

