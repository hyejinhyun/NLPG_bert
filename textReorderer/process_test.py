from nltk.tokenize import sent_tokenize
from itertools import permutations

import pandas as pd
import csv
import random

# input_file = dev_nsp.tsv
def preprocess(input_file):
    dataset = pd.read_csv(input_file, delimiter='\t')

    sentences = []
    for c in dataset.context:
        sentences.append(c)

    count = 0
    test_result = []

    sentence0 = []
    sentence1 = []
    sentence2 = []
    sentence3 = []

    # test용 preprocess
    for s in sentences:
        sentence_list = sent_tokenize(s)
        length = len(sentence_list)
        index = length // 4

        if length % 4 == 0:
            sentence0 = sentence_list[0:index]
            sentence1 = sentence_list[index:index * 2]
            sentence2 = sentence_list[index * 2:index * 3]
            sentence3 = sentence_list[index * 3:]

        elif length % 4 == 1:
            sentence0 = sentence_list[0:index]
            sentence1 = sentence_list[index:index * 2]
            sentence2 = sentence_list[index * 2:index * 3]
            sentence3 = sentence_list[index * 3:]

        elif length % 4 == 2:
            sentence0 = sentence_list[0:index]
            sentence1 = sentence_list[index:index * 2]
            sentence2 = sentence_list[index * 2:index * 3 + 1]
            sentence3 = sentence_list[index * 3 + 1:]

        elif length % 4 == 3:
            sentence0 = sentence_list[0:index]
            sentence1 = sentence_list[index:index * 2 + 1]
            sentence2 = sentence_list[index * 2 + 1:index * 3 + 1]
            sentence3 = sentence_list[index * 3 + 1:]

        s0 = ''.join(sentence0)
        s1 = ''.join(sentence1)
        s2 = ''.join(sentence2)
        s3 = ''.join(sentence3)

        count = sentences.index(s)
        test = []
        items = [s1, s2, s3]

        for i in list(permutations(items, 3)):
            a = (s0,)
            c = a + i
            test.append(c)

        # random()으로 리스트 원소 중 하나
        result_tuple = random.choice(test)
        result = list(result_tuple)

        test_result.append([])
        for i in result:
            test_result[count].append(i)

        count += 1

    return test_result



