# -*-coding:UTF-8 -*-
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import torch.nn.functional as F

import pandas as pd
import csv

def toBertIds(question_input):
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question_input)))

def output_write(output_predict_file, predicted_token, logit_probability, answer, sentence):
    # PREDICTED_TOKEN_INDEX = 0
    # LOGIT_PROB_INDEX = 1
    # ANSWER_TOKEN_INDEX = 2
    # SENTENCE_INDEX = 3

    with open(output_predict_file, 'at', -1, "utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow([predicted_token, logit_probability, answer, sentence])

def convert_data_to_context(filepath):
    dataset = pd.read_csv(filepath, delimiter='\t')
    
    sentences = []
    for c in dataset.text:
        sentences.append(c)

    return sentences

def get_answer(filepath):
    dataset = pd.read_csv(filepath, delimiter='\t')
    
    sentences = []
    for c in dataset.answer:
        sentences.append(c)

    return sentences


if __name__ == "__main__":

    # load and init
    tokenizer = BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
    
    bert_config, bert_class = (BertConfig, BertForMaskedLM)
    config = bert_config.from_pretrained('trained_model/2/config.json')
    model = bert_class.from_pretrained('trained_model/2/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-uncased'), config=config)
    model.eval()

    question_inputs = convert_data_to_context('mlm_test.tsv')
    question_answers = get_answer('mlm_test.tsv')


    for question_input in question_inputs:
        # input_str = ' '.join(question_input)
        # tokenized_text = tokenizer.tokenize(input_str)
        index = question_inputs.index(question_input)
        tokenized_text = tokenizer.tokenize(question_input)
        
        if '[MASK]' not in tokenized_text:
            continue

        maskpos = tokenized_text.index('[MASK]')
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        outputs = model(tokens_tensor)
        predictions = outputs[0]

        logit_prob = F.softmax(predictions[0, maskpos]).data.tolist()
        predicted_index = torch.argmax(predictions[0, maskpos]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print(predicted_token,logit_prob[predicted_index])
        output_write("blank_test_results.tsv", predicted_token, logit_prob[predicted_index], question_answers[index], question_inputs[index])