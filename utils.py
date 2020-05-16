import torch
import random
import math
import collections
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F


import pandas as pd

device = torch.device("cpu")

def format_special_chars(tokens):
    return [t.replace('Ġ', '').replace('▁', '').replace('</w>', '') for t in tokens]

def format_attention(attention, tokens):
    """ Set special token <sep>, <cls> attention to zero and format the attention """
    # set special token's attention to zero
    for i, t in enumerate(tokens):
        if t in ("<sep>", "<cls>"):
            for layer_attn in attention:
                layer_attn[0, :, i, :] = 0
                layer_attn[0, :, :, i] = 0
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("Wrong attention length, attention length must be 4")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def look_score(attn_data, index_a, index_b):
    """ Look pair attention score in layers, head """
    score = 0.
    for layer in attn_data:
        for head in layer:
            score_individual = head[index_a][index_b].tolist()
            score += score_individual
    return round(score, 3)

def pair_match(sentence_a_tokens, sentence_b_tokens, attn_data=None):
    """ Matching each token in sentence_a and sentence_b and making pairs """
    pairs = []
    for index_a in range(len(sentence_a_tokens)):
        for index_b in range(len(sentence_b_tokens)):
            if attn_data is not None:
                score = look_score(attn_data, index_a, index_b)
                pair = (sentence_a_tokens[index_a], sentence_b_tokens[index_b], score)
                # filter the special token
                if score != 0:
                    pairs.append(pair)
            else:
                # for evaluation pairs
                pair = (sentence_a_tokens[index_a], sentence_b_tokens[index_b])
                pairs.append(pair)
    return pairs

def pair_match_span_detection(sentence_a_tokens, sentence_b_tokens, attn_data, top_percent):
    """ Matching each token in sentence_a and sentence_b and making pairs """
    pairs = []
    for index_a in range(len(sentence_a_tokens)):
        for index_b in range(len(sentence_b_tokens)):
            score = look_score(attn_data, index_a, index_b)
            pair = (index_a, score)
            # filter the special token
            if score != 0:
                pairs.append(pair)
    pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    topk_slice = math.floor((len(pairs)/100)*top_percent)
    pairs = pairs[:topk_slice]
    tokens_count = []
    for index, score in pairs:
        tokens_count.append(index)
    counter = collections.Counter(tokens_count)
    return counter.most_common(10)

def pair_without_score(pair):
    """ Return pairs without score """
    pairs = []
    for token_a, token_b, score in pair:
        if token_a != '' and token_b != '':
            pair = (token_a, token_b)
            pairs.append(pair)
    return pairs

def unique_pair_without_score(pair):
    pairs = []
    if len(pair[0]) == 3:
        for token_a, token_b, score in pair:
            if token_a != '' and token_b != '' and token_a not in pairs:
                pairs.append(token_a)
    else:
        for token_a, token_b in pair:
            if token_a != '' and token_b != '' and token_a not in pairs:
                pairs.append(token_a)
    return pairs

def MRR_calculate(pair_truth, pair_all):
    final_score = 0.
    for query in pair_truth:
        for response in range(len(pair_all)):
            if pair_all[response] == query:
                score = 1/(response+1)
                final_score += score
    final_score = final_score/len(pair_truth)
    return final_score

def MRR_mean(pair_truth, pair_all, top_k, times):
    """ Choose k tokens from tokens list for calculating MRR"""
    # Handle when the pairs number is lower than top_k situation
    if len(pair_truth) < top_k:
        top_k = len(pair_truth)
    filtered = random.choices(pair_truth, k=top_k)
    final = 0.
    for i in range(times):
        score = MRR_calculate(filtered, pair_all)
        final += score
    final = final/times
    return final

def pair_match_accumulation(sentence_a_tokens, sentence_b_tokens, attn_data):
    scores = []
    for index_a, text in enumerate(sentence_a_tokens):
        score_total = 0
        for index_b, text in enumerate(sentence_b_tokens):
            score = look_score(attn_data, index_a, index_b)
            score_total += score
        scores.append(score_total)
    return scores

def attention_weight_span(data, feature, output):

    input_ids = feature[0].input_ids
    tokens = feature[0].tokens
    token_type_ids = feature[0].token_type_ids
    
    attention = output[5]
    attn = format_attention(attention, tokens)
    
    sentence_b_start = token_type_ids.index(1)
    slice_a = slice(0, sentence_b_start)
    slice_b = slice(sentence_b_start, len(tokens))
    #head_slice = slice(0, 4)
    attn_data = attn[:, :, slice_a, slice_b]
    sentence_a_tokens = tokens[slice_a]
    sentence_b_tokens = tokens[slice_b]
    top_tokens = pair_match_span_detection(sentence_a_tokens, sentence_b_tokens, attn_data, 1)
    top_indexs = [pair[0] for pair in top_tokens]
    top_probs = [pair[1] for pair in top_tokens]
    top_n = len(top_indexs)

    log_probs = torch.tensor(top_probs, dtype=torch.float)
    start_top_log_probs = F.softmax(log_probs, dim=-1)
    end_top_log_probs = F.softmax(log_probs, dim=-1)
    end_top_log_probs = end_top_log_probs.unsqueeze(0).expand(top_n, -1).reshape(1, top_n*top_n)
    start_top_index = torch.tensor(top_indexs)
    end_top_index = torch.tensor(top_indexs)
    end_top_index = end_top_index.unsqueeze(0).expand(top_n, -1).reshape(1, top_n*top_n)
    cls_logits = 0
    
    return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits, top_n)

def explainability_compare(model, tokenizer, sentence_a, sentence_b, test_sentence_a, unique=False, in_un=False, top_k=None):
    """ Evaluating MRR between model and attention span"""
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids'].to(device)
    input_ids.squeeze()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    token_type_ids = inputs['token_type_ids'].to(device)
    
    model.eval()
    with torch.no_grad():
        attention = model(input_ids, token_type_ids=token_type_ids, task=torch.tensor([0]))[-1]
    
    attn = format_attention(attention, tokens)  
    tokens = format_special_chars(tokens)
    sentence_b_start = token_type_ids[0].tolist().index(1)
    slice_a = slice(0, sentence_b_start)
    slice_b = slice(sentence_b_start, len(tokens))
    attn_data = attn[:, :, slice_a, slice_b]
    sentence_a_tokens = tokens[slice_a]
    sentence_b_tokens = tokens[slice_b]
    pair = pair_match(sentence_a_tokens, sentence_b_tokens, attn_data=attn_data)
    pair = sorted(pair, key=lambda pair: pair[2], reverse=True)
    if not unique:
        pair = pair_without_score(pair)
    else:
        pair = unique_pair_without_score(pair)
    
    test_inputs = tokenizer.encode_plus(test_sentence_a, sentence_b, return_tensors='pt', add_special_tokens=False)
    test_input_ids = test_inputs['input_ids']
    test_input_ids.squeeze()
    test_tokens = tokenizer.convert_ids_to_tokens(test_input_ids.squeeze().tolist())
    test_token_type_ids = test_inputs['token_type_ids']
    test_tokens = format_special_chars(test_tokens)
    test_sentence_b_start = test_token_type_ids[0].tolist().index(1)
    test_slice_a = slice(0, test_sentence_b_start)
    test_slice_b = slice(test_sentence_b_start, len(test_tokens))
    test_sentence_a_tokens = test_tokens[test_slice_a]
    test_sentence_b_tokens = test_tokens[test_slice_b]
    test_pair = pair_match(test_sentence_a_tokens, test_sentence_b_tokens, attn_data=None)
    if unique or in_un:
        test_pair = unique_pair_without_score(test_pair)

    if top_k:
        score = MRR_mean(test_pair, pair, top_k=top_k, times=top_k)
    elif in_un:
        score = intersect_union(test_pair, pair)
    else:
        score = MRR_calculate(test_pair, pair)

    return score, len(test_pair)

""" Data Related Functions """

class Dataset_MRR(Dataset):
    """ RTE 3way dataset """
    
    def __init__(self, mode, tokenizer):
        assert mode in ["test_MRR"]
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        text_a, text_b, text_eval, label, t3, t5, t7 = self.df.iloc[idx, :].values
        label_tensor = torch.tensor(label)
            
        # sentence_a tokens
        word_pieces = []
        tokens_a = self.tokenizer.tokenize(text_a + '<SEP>')
        word_pieces += tokens_a
        len_a = len(tokens_a)
        
        # sentence_b tokens
        tokens_b = self.tokenizer.tokenize(text_b + '<SEP><CLS>')
        word_pieces += tokens_b
        len_b = len(word_pieces) - len_a
        
        # 將 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * (len_b-1) + [2], 
                                        dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor, text_a, text_b, text_eval)
    
    def __len__(self):
        return self.len

def MRR_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    text_a = [s[3] for s in samples]
    text_b = [s[4] for s in samples]
    
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1，讓 model 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids, text_a, text_b