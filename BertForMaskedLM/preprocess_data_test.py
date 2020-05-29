# -*-coding:UTF-8 -*-
from transformers import BertTokenizer
import json
import random
import torch
from torch.utils.data import TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('stopwords')

def LoadJson(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        AllData = json.load(f)
    return AllData


def convert_data_to_context(filepath, dataset):
    DRCD = LoadJson(filepath)
    tokenizer = BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')

    context_tokens = []
    context_loss_tokens = []
    sample = []

    # BertForMaskedLM
    for data in DRCD["data"]:
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            little_context = context[:128]
            sample.append(little_context)

    index = round(len(sample)*0.25)
    if dataset == "test1":
        small_sample = sample[:index]
    elif dataset == "test2":
        small_sample = sample[index:index*2]
    elif dataset == "test3":
        small_sample = sample[index*2:index*3]
    else:
        small_sample = sample[index*3:]

    for c in small_sample:
        c_c = conversion_context(c, tokenizer, context_loss_tokens)
        context_tokens.append(c_c)

    # print(context_tokens)

    return context_tokens

def conversion_context(context, tokenizer, context_loss_tokens):
    Mask_id_list = []
    new_word_piece_list = []
    context_tokens=[]

    word_piece_list = tokenizer.tokenize(context)
    result = TextRank().get_keyword(context)
    keyword = result[0]

    context_tokens = random_change_word_piece(tokenizer, word_piece_list, Mask_id_list, keyword, new_word_piece_list)

    return context_tokens


def random_change_word_piece(tokenizer, word_piece_list, Mask_id_list, k, new_word_piece_list):
    count = 0
    new_word_piece_list.append('[CLS]')
    for word_piece in word_piece_list:
        if word_piece == k[0]:
            Mask_id_list.append(tokenizer.convert_tokens_to_ids(word_piece))
            new_word_piece_list.append('[MASK]')
        else:
            new_word_piece_list.append(word_piece)
    new_word_piece_list.append('[SEP]')

    return new_word_piece_list



class SentenceTokenizer(object):
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    def get_tokens(self, sentences):
        # tokens = tokenizer.tokenize(sentences)  # word_piece_list

        tokens = word_tokenize(sentences)

        # 이건 불용어 빼주는 작업
        tr_tk = []

        for w in tokens:
            if w not in self.stopwords:
                tr_tk.append(w)

        return tr_tk

class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []

    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}


class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1

        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}


class TextRank(object):
    def get_keyword(self, text, word_num=1):
        tokenizer = BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')

        tokens = SentenceTokenizer().get_tokens(text)
        words_graph, idx2word = GraphMatrix().build_words_graph(tokens)

        rank = Rank()
        rank_idx = rank.get_ranks(words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)

        keywords = []
        index = []
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)

        for idx in index:
            # keywords.append(idx2word[idx])
            t = tokenizer.tokenize(idx2word[idx])
            keywords.append(t)


        # print(keywords)
        return keywords
