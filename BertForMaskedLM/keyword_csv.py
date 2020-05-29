# -*-coding:UTF-8 -*-
from keyword_list import convert_data_to_context
# from keyword import *

# import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import torch.nn.functional as F

import csv

def toBertIds(question_input):
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question_input)))

def output_write(output_predict_file, keyword):
    # PREDICTED_TOKEN_INDEX = 0
    # LOGIT_PROB_INDEX = 1

    with open(output_predict_file, 'at', -1, "utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow([keyword])

if __name__ == "__main__":

    # load and init
    tokenizer = BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
    
    bert_config, bert_class = (BertConfig, BertForMaskedLM)
    config = bert_config.from_pretrained('trained_model/0/config.json')
    model = bert_class.from_pretrained('trained_model/0/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-uncased'), config=config)
    model.eval()

    keyword1 = convert_data_to_context('dev-v2.0.json', 'test1')
    keyword2 = convert_data_to_context('dev-v2.0.json', 'test2')
    keyword3 = convert_data_to_context('dev-v2.0.json', 'test3')
    keyword4 = convert_data_to_context('dev-v2.0.json', 'test4')

    for k in keyword1:
        output_write("keyword.tsv", k)

    for k in keyword2:
        output_write("keyword.tsv", k)

    for k in keyword3:
        output_write("keyword.tsv", k)

    for k in keyword4:
        output_write("keyword.tsv", k)