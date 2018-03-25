# coding:utf-8
import os
import logging
import pickle
from collections import Counter
import itertools as it

import numpy as np
import pymorphy2
import string
morph = pymorphy2.MorphAnalyzer()

from gensim.models import KeyedVectors, Word2Vec
import nltk
from nltk import word_tokenize

from sklearn.neighbors import BallTree

nltk.download('stopwords')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

wv_file = 'wv_keyed_vectors.w2v'
corpus_file = 'ivi_corpus.pkl'
d2v_file = 'doc2vec.pkl'
item_to_qid_file = 'item2qid.pkl'
content_descr_file = 'content_descr.tsv'
doc_vecs = 'content_vecs.pkl'
array_file = 'content_array.pkl'
vectorized_file = 'corpus_vectors.pkl'
MIN_TOKEN_COUNT = 1

def tokenize_tsv(content_descr_file, tokenized_corpus_file):
    """Разбиение текстовых строк на токены
    Каждый токен приводим к нормальной форме

    :param content_descr_file: tsv с текстовым описанием контента
    :param tokenized_corpus_file: файл, в который дампим токены
    :return: dict {obj_id: tokens}
    """
    if os.path.isfile(tokenized_corpus_file):
        corpus = pickle.load(open(tokenized_corpus_file, 'rb'))
    else:
        txt = open(content_descr_file, 'r').readlines()
        corpus = dict()
        n = 0
        for t in txt:
            n += 1
            words = np.array(word_tokenize(t))
            if n % 1000 == 0:
                logger.info("doc {}".format(n))
            corpus.update({words[0]: [
                morph.parse(str(i).lower())[0].normal_form for i in words[1:]
                if (i not in list(string.punctuation) + ['«', '»', '–']
                    and i not in nltk.corpus.stopwords.words('russian'))
            ]})
    corpus = filter_corpus(corpus, min_count=MIN_TOKEN_COUNT)
    pickle.dump(corpus, open(tokenized_corpus_file, 'wb'), protocol=2)
    return corpus

def filter_corpus(corpus, min_count = 5):
    """Удаляем низкочастотные слова из корпуса"""
    vocab_counter = Counter(it.chain(*corpus.values()))
    logger.info('Filter rare words')
    for k in corpus:
        arr = corpus[k]
        corpus[k] = [m for m in arr if vocab_counter[m] >= min_count]
    return corpus

def vectorize_tokens(corpus, wv_file):
    """Обучение Word2Vec"""
    if os.path.isfile(wv_file):
        word_vectors = KeyedVectors.load(wv_file)
    else:
        model = Word2Vec(
            sentences=corpus.values(), size=100, window=5,
            min_count=MIN_TOKEN_COUNT, workers=10)
        word_vectors = model.wv
        del model
    word_vectors.save(wv_file)
    return word_vectors

def group_by_index(prepared_text_corpus, item2qid):
    """Собираем серии для каждого сериала в один документ"""
    grouped_dict = dict()
    for k in prepared_text_corpus:
        raw_tokens = prepared_text_corpus[k]
        if isinstance(raw_tokens, np.float64):
            raw_tokens = np.zeros(100)
        group_key = item2qid.get(int(k), int(k))
        if doc2vec.get(group_key) is None:
            grouped_dict[group_key] = raw_tokens
        else:
            grouped_dict[group_key] = np.hstack(grouped_dict[group_key], raw_tokens)
    return grouped_dict

def vectorize_doc(prepared_text_corpus, word_vectors, d2v_file):
    if os.path.isfile(d2v_file):
        doc2vec = pickle.load(open(d2v_file, 'rb'))
    else:
        doc2vec = dict()
        for k in prepared_text_corpus:
            raw_tokens = prepared_text_corpus[k]
            token_vectors = np.array([word_vectors[t] for t in raw_tokens])
            d2v = token_vectors.mean(axis=0)
            doc2vec[k] = np.zeros(100) if isinstance(d2v, np.float64) else d2v
    pickle.dump(doc2vec, open(d2v_file, 'wb'), protocol=2)
    return doc2vec

def prepare_doc(corpus_file, content_descr_file, vectorized_corpus_file):
    """Векторизуем TSV файл

    :return:
    """
    tokenized_corpus = tokenize_tsv(content_descr_file, corpus_file)
    word_vectors = vectorize_tokens(tokenized_corpus, wv_file)

    logger.info("любовь {}".format(word_vectors.most_similar('любовь')))
    logger.info("Векторизуем корпус текстов")
    vectorized_doc = vectorize_doc(tokenized_corpus, word_vectors, vectorized_corpus_file)
    return vectorized_doc

def make_query(search_tree, test_vector, docs):
    dist, ind = search_tree.query(test_vector.reshape(1, -1), k=3)
    search_result = [docs[i] for i in ind[0]]
    # индекс контента
    qids = [int(k) for k in search_result]
    return qids

def dict2array(corpus_dict, content_vecs_file):
    if os.path.isfile(content_vecs_file):
        content_data = pickle.load(open(content_vecs_file, 'rb'))
    else:
        content_data = np.vstack([
            np.zeros(100) if isinstance(i, np.float64) else i
            for i in list(corpus_dict.values())
        ])

        pickle.dump(content_data, open(content_vecs_file, 'wb'), protocol=2)
    return content_data, list(corpus_dict.keys())

logger.info("Загружаем индекс документов")
doc2vec = prepare_doc(corpus_file, content_descr_file, vectorized_file)
logger.info("Загружаем векторы слов")
word_vectors = KeyedVectors.load(wv_file)
# группируем контент из одиночных серий в сериалы
item2qid = pickle.load(open(item_to_qid_file, 'rb'))
logger.info('Корпус документов: {}'.format(len(list(doc2vec.keys()))))
grouped_docs = group_by_index(doc2vec, item2qid)
grouped_docs_keys = list(grouped_docs.keys())
logger.info('Сгруппированные документы: {}'.format(len(grouped_docs_keys)))
logger.info("Формируем индекс контента")
content_data, content_qids = dict2array(grouped_docs, array_file)
# объект для поиска
nn_search = BallTree(content_data, leaf_size=40, metric='minkowski')
test_vector = word_vectors['любовь']
qids = make_query(nn_search, test_vector, content_qids)
logger.info("Кинчики о любви {}".format(qids))