# -*-coding:UTF-8 -*-
from preprocess_data_test import convert_data_to_context

import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import torch.nn.functional as F

import csv

def toBertIds(question_input):
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question_input)))

def output_write(output_predict_file, predicted_token, logit_probability):
    # PREDICTED_TOKEN_INDEX = 0
    # LOGIT_PROB_INDEX = 1

    with open(output_predict_file, 'at', -1, "utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow([predicted_token, logit_probability])

if __name__ == "__main__":

    # load and init
    tokenizer = BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
    
    bert_config, bert_class = (BertConfig, BertForMaskedLM)
    config = bert_config.from_pretrained('trained_model/0/config.json')
    model = bert_class.from_pretrained('trained_model/0/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-uncased'), config=config)
    model.eval()

    question_inputs1 = convert_data_to_context('dev-v2.0.json', 'test1')
    question_inputs2 = convert_data_to_context('dev-v2.0.json', 'test2')
    question_inputs3 = convert_data_to_context('dev-v2.0.json', 'test3')
    question_inputs4 = convert_data_to_context('dev-v2.0.json', 'test4')

    for question_input in question_inputs1:
        input_str = ' '.join(question_input)
        tokenized_text = tokenizer.tokenize(input_str)
        
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
        output_write("test_results.tsv", predicted_token, logit_prob[predicted_index])

    for question_input in question_inputs2:
        input_str = ' '.join(question_input)
        tokenized_text = tokenizer.tokenize(input_str)

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
        output_write("test_results.tsv", predicted_token, logit_prob[predicted_index])

    for question_input in question_inputs3:
        input_str = ' '.join(question_input)
        tokenized_text = tokenizer.tokenize(input_str)

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
        output_write("test_results.tsv", predicted_token, logit_prob[predicted_index])

    for question_input in question_inputs4:
        input_str = ' '.join(question_input)
        tokenized_text = tokenizer.tokenize(input_str)

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
        output_write("test_results.tsv", predicted_token, logit_prob[predicted_index])
