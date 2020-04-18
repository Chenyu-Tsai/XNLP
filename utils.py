import torch

def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("Wrong attention length, attention length must be 4")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def look_score(attn_data, index_a, index_b):
    score = 0.
    for layer in attn_data:
        for head in layer:
            score_individual = head[index_a][index_b].tolist()
            score += score_individual
    return round(score, 3)

def pair_match(sentence_a_tokens, sentence_b_tokens):
    whole = []
    for token_a in sentence_a_tokens:
        index_a = sentence_a_tokens.index(token_a)
        for token_b in sentence_b_tokens:
            index_b = sentence_b_tokens.index(token_b)
            score = look_score(attn_data, index_a, index_b)
            pair = (token_a, token_b, score)
            if score != 0:
                whole.append(pair)
    return whole
