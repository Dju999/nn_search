# coding:utf-8
import os
import logging
import pickle
from collections import Counter
import itertools as it
import re

import numpy as np
import pymorphy2
import string
morph = pymorphy2.MorphAnalyzer()

from gensim.models import KeyedVectors, Word2Vec
import nltk
from nltk import word_tokenize
# для tf-idf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pysparnn.cluster_index as ci

from sklearn.neighbors import BallTree

nltk.download('stopwords')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

wv_file = 'wv_keyed_vectors.w2v'
corpus_file = 'ivi_corpus_tokenized.pkl'
d2v_file = 'doc2vec.pkl'
item_to_qid_file = 'item2qid.pkl'
content_descr_file = 'content_descr.tsv'
doc_vecs = 'content_vecs.pkl'
array_file = 'content_array.pkl'
vectorized_file = 'corpus_vectors.pkl'
MIN_TOKEN_COUNT = 1

config = {
    "working_dir": 'data',
    "content_description_raw": 'content_descr.tsv',
    "content_descr_tokens": 'ivi_corpus_tokenized.pkl',
    "content_descr_vectors": 'ivi_corpus_vectorized.pkl',
    "word_vectors_file": 'wv_keyed_vectors.w2v',
    "item_to_qid_file": 'item2qid.pkl',
    'content_array_file': 'content_array_file'
}


class Content2Vec(object):
    """Получаем эмбеддинг контента из текстового описани
    """

    def __init__(self, config):
        self.wd = config["working_dir"]
        if not os.path.isdir('data'):
            os.mkdir(self.wd)
        self.morph = pymorphy2.MorphAnalyzer()
        self.content_descr_raw = config["content_description_raw"]
        self.item_to_qid_file = config["item_to_qid_file"]
        self.content_descr_tokens = os.path.join(self.wd, config["content_descr_tokens"])
        self.word_vectors_file = os.path.join(self.wd, config["word_vectors_file"])
        self.content_vectors_file = os.path.join(self.wd, config["content_descr_vectors"])
        self.content_array_file = os.path.join(self.wd, config["content_array_file"])
        self.item2qid = pickle.load(open(self.item_to_qid_file, 'rb'))
        self.tokenized_corpus = None
        self.word_vectors = None
        self.content_vectors = None
        self.content_data = None
        self.content_qids = None
        self.content_vectors_array = None
        self.content_index_array = None
        self.search_index_wv = None
        self.cnt_vectorizer = CountVectorizer()
        self.tf_idf = TfidfTransformer(smooth_idf=False)
        self.tf_idf_features = None
        self.search_index_tfidf = None

    def build(self):
        """Инициализируем поля класса
        
        :return: 
        """
        # получаем эмбеддинги контента и токенов
        self.content_to_vec()
        self.build_search_index()

    def content_to_vec(self):
        """Получаем векторное описание для каждого контента
        
        :return: 
        """
        # конвертируем описание контента в токены
        self.tokenize_tsv()
        # для каждого токена получаем векторное описание с помощью Word2Vec
        self.vectorize_tokens()
        logger.info("планета {}".format(self.word_vectors.most_similar('планета')))
        logger.info("Векторизуем корпус текстов")
        self.vectorize_content()

    def tokenize_tsv(self):
        """Разбиение текстовых строк на токены
        Каждый токен приводим к нормальной форме. Группируем контент в сборники

        :return: dict {obj_id: tokens}
        """
        if os.path.isfile(self.content_descr_tokens):
            self.tokenized_corpus = pickle.load(open(self.content_descr_tokens, 'rb'))
        else:
            txt = open(self.content_descr_raw, 'r').readlines()
            self.tokenized_corpus = dict()
            n = 0
            for t in txt:
                n += 1
                content_id, content_descr = t.split('\t')
                if n % 1000 == 0:
                    logger.info("doc {}".format(n))
                self.tokenized_corpus.update({int(content_id): self.str2tokens(content_descr)})
            self.tokenized_corpus = self.filter_corpus(min_count=MIN_TOKEN_COUNT)
            # объединяем эмбеддинги серий в эмбеддинг сериала
            self.tokenized_corpus = self.group_compilations()
            pickle.dump(self.tokenized_corpus, open(self.content_descr_tokens, 'wb'), protocol=2)
        # print('tokenize_csv()', self.tokenized_corpus)
        #print(list(self.tokenized_corpus.keys())[:10])
        #print(self.tokenized_corpus[1092450])
        return self.tokenized_corpus

    def token2normal(self, raw_token):
        """Превращаем строку в набор токенов
        
        :param raw_string: 
        :return: 
        """
        return self.morph.parse(str(raw_token).lower())[0].normal_form

    def str2tokens(self, raw_string):
        words = np.array(word_tokenize(raw_string.replace('.', ' ')))
        return [
            self.token2normal(i) for i in words
            if (i not in list(string.punctuation) + ['«', '»', '–']
                and i not in nltk.corpus.stopwords.words('russian'))
        ]

    def tokens2vec(self, raw_tokens, engine='w2v'):
        agg_tokens_vector = []
        if engine == 'w2v':
            token_vectors = np.array([self.word_vectors[t] for t in raw_tokens])
            # вектор контента - это усреднение векторов его токенов
            agg_tokens_vector = token_vectors.mean(axis=0)
        elif engine == 'tfidf':
            agg_tokens_vector = self.cnt_vectorizer.transform([' '.join(raw_tokens)]).toarray()
        return agg_tokens_vector

    def filter_corpus(self, min_count):
        """Удаляем низкочастотные слова из корпуса"""
        vocab_counter = Counter(it.chain(*self.tokenized_corpus.values()))
        logger.info('Filter rare words')
        for k in self.tokenized_corpus:
            arr = self.tokenized_corpus[k]
            filtered_arr = np.array([m for m in arr if vocab_counter[m] >= min_count])
            self.tokenized_corpus[k] = filtered_arr if len(filtered_arr) > 0 else np.array(['empty'])
        return self.tokenized_corpus

    def vectorize_tokens(self):
        """Обучение Word2Vec"""
        if os.path.isfile(self.word_vectors_file):
            self.word_vectors = KeyedVectors.load(self.word_vectors_file)
        else:
            # print("tokenized corpus \n", list(self.tokenized_corpus.values()))
            model = Word2Vec(
                sentences=list(self.tokenized_corpus.values()), size=100, window=5,
                min_count=MIN_TOKEN_COUNT, workers=5)
            self.word_vectors = model.wv
            del model
            self.word_vectors.save(self.word_vectors_file)
        # print(self.word_vectors.wv.vocab)
        return self.word_vectors

    def vectorize_content(self):
        # строим матрицу tf-idf
        text_corpus = [' '.join(list(k)) for k in self.tokenized_corpus.values()]
        sparse_token_counters = self.cnt_vectorizer.fit_transform(text_corpus)
        self.tf_idf_features = self.tf_idf.fit_transform(sparse_token_counters)
        self.tf_idf_features.astype(np.float32)
        print("tf-idf features ", self.tf_idf_features.toarray().shape)
        # усредняем векторы Word2Vec
        if os.path.isfile(self.content_vectors_file):
            self.content_vectors = pickle.load(open(self.content_vectors_file, 'rb'))
        else:
            self.content_vectors = dict()
            for k in self.tokenized_corpus:
                raw_tokens = self.tokenized_corpus[k]
                current_content_vector = self.tokens2vec(raw_tokens, engine='w2v')
                self.content_vectors[k] = (
                    np.zeros(100).tolist()
                    if isinstance(current_content_vector, np.float64)
                    else current_content_vector.tolist()
                )
            pickle.dump(self.content_vectors, open(self.content_vectors_file, 'wb'), protocol=2)
        return self.content_vectors

    def group_compilations(self):
        """Собираем серии для каждого сериала в один вектор"""
        grouped_dict = dict()
        for k in self.tokenized_corpus:
            raw_tokens = self.tokenized_corpus[k]
            if isinstance(raw_tokens, np.float64):
                raw_tokens = np.zeros(100).tolist()
            group_key = self.item2qid[int(k)]
            if grouped_dict.get(group_key) is None:
                grouped_dict[group_key] = raw_tokens.tolist()
            else:
                grouped_dict[group_key] = np.hstack([grouped_dict[group_key], raw_tokens]).tolist()
        return grouped_dict

    def dict2array(self):
        self.content_vectors_array = np.vstack([
            np.zeros(100) if isinstance(i, np.float64) else i
            for i in list(self.content_vectors.values())
        ])
        pickle.dump(self.content_vectors_array, open(self.content_array_file, 'wb'), protocol=2)
        self.content_index_array = np.array(list(self.content_vectors.keys()), dtype=np.uint32)
        return self.content_vectors_array

    def build_search_index(self):
        """Формируем индекс для поиска по контенту"""
        # преобразуем в матрицу - для поиска ближайших соседей
        self.dict2array()
        logger.info("Строим поисковый индекс")
        # self.search_index_wv = BallTree(self.content_vectors_array, leaf_size=40, metric='braycurtis')
        # для tf-idf chebyshev, euclidean
        # self.search_index_tfidf = BallTree(self.tf_idf_features.todense(), leaf_size=40, metric='manhattan')
        # либа от FacebookResearch
        print('shape ftrs {}, content_index_array {}'.format(
            self.tf_idf_features.shape, self.content_index_array.shape))
        self.search_index_tfidf = ci.MultiClusterIndex(self.tf_idf_features.todense(), np.arange(self.content_index_array.size))
        logger.info("Построили индекс tf-idf")

    def make_query(self, query_str, engine='w2v'):
        query_tokens = self.str2tokens(query_str)
        print(query_tokens)
        query_vec = self.tokens2vec(query_tokens, engine=engine)
        if engine == 'w2v':
            dist, ind = self.search_index_wv.query(query_vec.reshape(1, -1), k=5)
        elif engine == 'tfidf':
            dist, ind = None, self.search_index_tfidf.search(query_vec, k=5, k_clusters=10, return_distance=False)
        print(dist, ind)
        search_result = [self.content_index_array[i] for i in ind[0]]
        # индекс контента
        qids = [int(k) for k in search_result]
        print("Контент по запросу {}: {}".format(query_str, qids))
        for i in qids:
            print('{}: {}'.format(i, self.tokenized_corpus[i]))
        return qids


if __name__ == '__main__':
    content2vec = Content2Vec(config)
    content2vec.build()
    # content2vec.make_query('Ирландия', engine='w2v')
    content2vec.make_query('Ирландия', engine='tfidf')
