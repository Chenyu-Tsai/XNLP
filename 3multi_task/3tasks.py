import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, RandomSampler
from transformers.modeling_utils import (WEIGHTS_NAME, PretrainedConfig, PreTrainedModel,
                             SequenceSummary, PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits)
from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetPreTrainedModel, XLNetModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data.dataset import ConcatDataset
from XLNet import (Dataset_3Way,
                  Dataset_multi, 
                  Dataset_Span_Detection, 
                  XLNetForMultiSequenceClassification, 
                  get_predictions)

import pandas as pd
import numpy as np
import random
from tqdm.notebook import tqdm, trange

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

trainset_3way = Dataset_3Way("RTE5_train", tokenizer=tokenizer, three_tasks=True)
trainset_multi = Dataset_multi("train_multi_label", tokenizer=tokenizer, three_tasks=True)
trainset_span = Dataset_Span_Detection("train_span_detection", tokenizer=tokenizer)
trainset = ConcatDataset([trainset_span, trainset_multi, trainset_3way])
train_sampler = RandomSampler(trainset)
train_dataloader = DataLoader(trainset, sampler=train_sampler, batch_size=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

PRETRAINED_MODEL_NAME = "xlnet-base-cased"
model = XLNetForMultiSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME,
                                                            output_attentions=True,
                                                            dropout=0.1)
model = model.to(device)

from torch.optim import AdamW

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

%%time
EPOCHS = 10
batch_size = 8
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=EPOCHS * ((len(train_dataloader)//batch_size)))
epochs_trained = 0

model.zero_grad()
train_iterator = trange(epochs_trained, EPOCHS, desc="Epoch")
set_seed(43)

for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    
    model.train()
    running_loss = 0.0
    batch_cnt = 1
    loss = torch.zeros(1).to(device)
    
    for step, data in enumerate(epoch_iterator):
        if data[0] == torch.tensor([0]) or data[0] == torch.tensor([1]):
            task = data[0]
            input_ids, token_type_ids, attention_mask, labels = [t.squeeze(0).to(device) for t in data[1:]]
            outputs = model(input_ids=input_ids, 
                            token_type_ids=token_type_ids, 
                            attention_mask=attention_mask, 
                            labels=labels,
                            task=task
                           )
        else:
            task = data[0]
            input_ids, attention_mask, token_type_ids, start_positions, end_positions, cls_index, p_mask = [t.squeeze(0).to(device) for t in data[1:]]
            outputs = model(input_ids=input_ids, 
                            token_type_ids=token_type_ids, 
                            attention_mask=attention_mask, 
                            start_positions=start_positions,
                            end_positions=end_positions,
                            cls_index=cls_index,
                            p_mask=p_mask,
                            task=task)
        batch_cnt += 1
        loss = outputs[0]/batch_size
        loss.backward()
        
        if batch_cnt >= batch_size:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            batch_cnt = 0

        # 紀錄當前 batch loss
        running_loss += loss.item()
    epochs_trained += 1
    
    testset = Dataset_3Way("RTE5_test", tokenizer=tokenizer, three_tasks=True)
    testloader = DataLoader(testset, batch_size=1)
    predictions = get_predictions(model, testloader)

    df_pred = pd.DataFrame({"label": predictions.tolist()})
        
    pred_Y = df_pred['label'].values
    test_Y = pd.read_csv("../data/RTE5_test.tsv", sep='\t').fillna("")['label'].values

    accuracy = accuracy_score(test_Y, np.array(pred_Y))
    precision = precision_score(test_Y, pred_Y, average='macro')
    recall = recall_score(test_Y, pred_Y, average='macro')
    fscore = f1_score(test_Y, pred_Y, average='macro')
    
    CNT = 0
    TOTAL = 0
    for i in range(len(test_Y)):
        if test_Y[i] == 2:
            TOTAL += 1
        else:
            pass
        if test_Y[i] == 2 and predictions[i] == 2:
            CNT += 1
    contra = round((CNT/TOTAL)*100,1)
    if contra > 20 and accuracy > 0.58:
        torch.save(model, "0513-3multi_%g, %g, %g.pkl" % (round(accuracy, 2), contra, epochs_trained))
    print("Accuracy: %g\tPrecision: %g\tRecall: %g\tF-score: %g Loss: %g" % (accuracy, precision, recall, fscore, running_loss))
    print(contra)
    print("------------------------------------------")