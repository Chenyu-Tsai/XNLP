import torch
import random
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

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
    filtered = random.choices(pair_truth, k=top_k)
    final = 0.
    for i in range(times):
        score = MRR_calculate(filtered, pair_all)
        final += score
    final = final/times
    return final

def explainability_compare(model, tokenizer, sentence_a, sentence_b, test_sentence_a, unique=False):
    """ Evaluating MRR between model and attention span"""
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids'].to(device)
    input_ids.squeeze()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    token_type_ids = inputs['token_type_ids'].to(device)
    
    model.eval()
    with torch.no_grad():
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    
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
    if unique:
        test_pair = unique_pair_without_score(test_pair)

    return MRR_calculate(test_pair, pair), len(test_pair)

""" Data Related Functions """

class Dataset_multi(Dataset):
    """ Multi label dataset"""

    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test", "train_RTE", "test_RTE", "train_multi", "test_multi", "train_multi_500"]
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        if self.mode == "train_multi":
            self.df = self.df[['text_a', 'text_b', 'labels']]
        self.len = len(self.df)
        self.label_map = {'True': 0, 'False': 1}
        self.tokenizer = tokenizer
    
    # return trainging data
    def __getitem__(self, idx):
        if self.mode in ["test", "train_RTE", "test_RTE", "test_multi", "test_2"]:
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b = self.df.iloc[idx, :2].values
            label = self.df.iloc[idx, 2].replace('[', '')
            label = label.replace(']','')
            label = np.fromstring(label, dtype=int, sep=',')
            #print(type(label.to_list()))
            label_tensor = torch.tensor(label, dtype=torch.float)
        
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
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

class Dataset_3Way(Dataset):
    """ RTE 3way dataset """
    
    def __init__(self, mode, tokenizer):
        assert mode in ["train_filtered", "test", "train"]
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {'ENTAILMENT': 0, 'UNKNOWN': 1, 'CONTRADICTION': 2}
        self.tokenizer = tokenizer
        
    
    # 回傳訓練資料的 function
    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, text_b, label = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label)
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label)
            
        # 第一句 tokens
        word_pieces = []
        tokens_a = self.tokenizer.tokenize(text_a + '<SEP>')
        word_pieces += tokens_a
        len_a = len(tokens_a)
        
        # 第二句 tokens
        tokens_b = self.tokenizer.tokenize(text_b + '<SEP><CLS>')
        word_pieces += tokens_b
        len_b = len(word_pieces) - len_a
        
        # 將 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * (len_b-1) + [2], 
                                        dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

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