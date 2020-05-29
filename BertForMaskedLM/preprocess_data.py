# -*-coding:UTF-8 -*-
from transformers import BertTokenizer
import json
import random
import torch
from torch.utils.data import TensorDataset
import itertools


def LoadJson(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        AllData = json.load(f)
    return AllData

def convert_data_to_feature(filepath, mode):

    DRCD = LoadJson(filepath)
    tokenizer = BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')

    context_tokens = []
    context_loss_tokens = []
    max_seq_len = 0

    # BertForMaskedLM
    for data in DRCD["data"]:
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            do = True

            while(len(context) >= 120): 
                little_context = context[:120]

                max_seq_len = conversion_context(little_context, tokenizer, max_seq_len, context_tokens, context_loss_tokens)
                context = context[60:]
                if len(context) <= 60:
                    do = False


            if do:
                max_seq_len = conversion_context(context, tokenizer, max_seq_len, context_tokens, context_loss_tokens)
    
    print("max length:",max_seq_len)   
    print(str(len(context_tokens)))
    assert max_seq_len <= 128


    for c in context_tokens:
        while len(c)<max_seq_len:
            c.append(0)

    for c_l in context_loss_tokens:
        while len(c_l)<max_seq_len:
            c_l.append(0)
            # c_l.append(-1)
    
    # BERT input embedding
    input_ids = context_tokens
    loss_ids = context_loss_tokens
    assert len(input_ids) == len(loss_ids)
    data_features = {'input_ids':input_ids,
                    'loss_ids':loss_ids}
    index = round(len(data_features) * 0.9)
    if mode=="train":
        return dict(itertools.islice(data_features.items(), index))
        # return data_features[:index]
    elif mode=="eval":
        return dict(itertools.islice(data_features.items(), len(data_features)-index, None))
        # return data_features[index:]



def conversion_context(context, tokenizer, max_seq_len, context_tokens, context_loss_tokens):
    Mask_id_list = []
    new_word_piece_list = []
    while len(Mask_id_list) == 0:
        word_piece_list = tokenizer.tokenize(context)

        random_change_word_piece(tokenizer, word_piece_list, Mask_id_list, new_word_piece_list)


    bert_ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(new_word_piece_list))

    bert_loss_ids = []
    # bert_loss_ids.append(-1)
    bert_loss_ids.append(0)
    Mask_id_count = 0
    for word_piece in new_word_piece_list:
        if word_piece == '[MASK]':
            try:
                bert_loss_ids.append(Mask_id_list[Mask_id_count])
                Mask_id_count = Mask_id_count + 1
            except :
                print(new_word_piece_list)
                print(Mask_id_count)
                print(Mask_id_list)
                assert Mask_id_count < len(Mask_id_list)
        else:
            bert_loss_ids.append(0)
            # bert_loss_ids.append(-1)

    # bert_loss_ids.append(-1)
    bert_loss_ids.append(0)

    context_tokens.append(bert_ids)
    context_loss_tokens.append(bert_loss_ids)
    assert len(bert_ids) == len(bert_loss_ids)

    if(len(bert_ids)>max_seq_len):
        return len(bert_ids)

    Mask_id_list.clear()
    new_word_piece_list.clear()
    word_piece_list.clear()
    return max_seq_len


def random_change_word_piece(tokenizer, word_piece_list, Mask_id_list, new_word_piece_list):
    count = 0
    for word_piece in word_piece_list:
        if 0.15 >= random.random():
            change_probability = random.random()
            if 0.8 >= change_probability:

                Mask_id_list.append(tokenizer.convert_tokens_to_ids(word_piece))
                new_word_piece_list.append('[MASK]')
            elif 0.9 >= change_probability:

                vocab_index = random.randint(0, len(tokenizer.vocab)-1)
                new_word_piece_list.append(tokenizer.convert_ids_to_tokens(vocab_index))
            else: 
                new_word_piece_list.append(word_piece)
        else:
            new_word_piece_list.append(word_piece)


    for new_word_piece in new_word_piece_list:
        if new_word_piece == '[MASK]':
            count = count + 1
    if count != len(Mask_id_list):
        Mask_id_list.clear()
        new_word_piece_list.clear()
        word_piece_list.clear()


def makeDataset(input_ids, loss_ids):
    all_input_ids = torch.tensor([input_id for input_id in input_ids], dtype=torch.long)
    all_loss_ids = torch.tensor([loss_id for loss_id in loss_ids], dtype=torch.long)
    return TensorDataset(all_input_ids, all_loss_ids)
        
if __name__ == "__main__":
    train_data_feature = convert_data_to_feature('train-v2.0.json', 'train')
    test_data_feature = convert_data_to_feature('train-v2.0.json', 'eval')
    train_dataset = makeDataset(input_ids=train_data_feature['input_ids'], loss_ids=train_data_feature['loss_ids'])
    test_dataset = makeDataset(input_ids=test_data_feature['input_ids'], loss_ids=test_data_feature['loss_ids'])

    # data_features = convert_data_to_feature('train-v2.0.json')
    # Dataset = makeDataset(data_features['input_ids'], data_features['loss_ids'])
   
