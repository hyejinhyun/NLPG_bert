# -*-coding:UTF-8 -*-
from preprocess_data import convert_data_to_feature, makeDataset
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, AdamW
# from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
import torch
import os
import torch.nn.functional as F     

# 動態調整學習率，參考網站:http://www.spytensor.com/index.php/archives/32/
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    # PreprocessData
    train_data_feature = convert_data_to_feature('train-v2.0.json', "train")
    test_data_feature = convert_data_to_feature('train-v2.0.json', "eval")
    train_dataset = makeDataset(input_ids = train_data_feature['input_ids'], loss_ids = train_data_feature['loss_ids'])
    test_dataset = makeDataset(input_ids = test_data_feature['input_ids'], loss_ids = test_data_feature['loss_ids'])
    train_dataloader = DataLoader(train_dataset,shuffle=False)
    test_dataloader = DataLoader(test_dataset ,batch_size=1 ,shuffle=True)

    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    Learning_rate = 5e-6     
    training_epoch = 2
    optimizer = AdamW(optimizer_grouped_parameters, lr=Learning_rate, eps=1e-8)
    # get_linear_schedule_with_warmup:https://github.com/huggingface/transformers/issues/1837
    # scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=5, t_total=training_epoch)

    for epoch in range(training_epoch):
        model.train()
        if epoch % 5 == 0 and epoch != 0:
            Learning_rate = Learning_rate * 0.5
            adjust_learning_rate(optimizer,Learning_rate)

        AllTrainLoss = 0.0
        count = 0
        for batch_index, batch_dict in enumerate(train_dataloader):
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(
                batch_dict[0],
                masked_lm_labels = batch_dict[1]
                )
            
            loss, logits = outputs[:2]
            AllTrainLoss += loss.item()
            count += 1

            model.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()  # Update learning rate schedule
            
        Average_train_loss = round(AllTrainLoss/count, 3)

        model.eval()
        AllTestLoss = 0.0
        count = 0
        for batch_index, batch_dict in enumerate(test_dataloader):
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(
                batch_dict[0],
                masked_lm_labels = batch_dict[1]
                )

            loss, logits = outputs[:2]
            AllTestLoss += loss.item()
            count += 1
        
        Average_test_loss = round(AllTestLoss/count, 3)

        print('pre' + str(epoch+1) + 'times' + 'train，loss:' + str(Average_train_loss) + '，eval，loss:' + str(Average_test_loss))

        folder = os.path.exists('trained_model/'+str(epoch))
        if not folder:
            os.makedirs('trained_model/'+str(epoch))
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained('trained_model/'+str(epoch))
    
