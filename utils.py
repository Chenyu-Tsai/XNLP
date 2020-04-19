import torch
import random

def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]

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
        pair = (token_a, token_b)
        pairs.append(pair)
    return pair

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

def explainability_compare(model, tokenizer, sentence_a, sentence_b, test_sentence_a, test_sentence_b):
    """ Evaluating MRR between model and attention span"""
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids'].cuda()
    input_ids.squeeze()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    token_type_ids = inputs['token_type_ids'].cuda()
    
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
    pair = pair_without_score(pair)
    
    test_inputs = tokenizer.encode_plus(test_sentence_a, test_sentence_b, return_tensors='pt', add_special_tokens=False)
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

    return MRR_calculate(test_pair, pair)
