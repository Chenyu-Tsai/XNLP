import argparse
from utils import explainability_compare
from transformers import XLNetTokenizer, XLNetPreTrainedModel

def calculate(model, dataloader, tokenizer):
    total = len(dataloader)
    entail_total = 0
    entail_total_len = 0
    neutral_total = 0
    neutral_total_len = 0
    contradict_total = 0
    contradict_total_len = 0
    
    entail_correct = 0
    entail_correct_len = 0
    neutral_correct = 0
    neutral_correct_len = 0
    contradict_correct = 0
    contradict_correct_len = 0
    
    entail_MRR_c = 0.
    neutral_MRR_c = 0.
    contradict_MRR_c = 0.
    
    entail_MRR_inc = 0.
    neutral_MRR_inc = 0.
    contradict_MRR_inc = 0.
    
    model.eval()
    with torch.no_grad():
        data_iterator = tqdm.tqdm(dataloader, desc='Iteration')
        for data in data_iterator:
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            # predict
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            sentence_a = data[4][0]
            sentence_b = data[5][0]
            eval_sentence = data[6][0]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            # divide 3 class
            label = data[3]
            MRR, length = explainability_compare(model, tokenizer, sentence_a, sentence_b, eval_sentence)

            if label == torch.tensor([0]):
                entail_total += 1
                entail_total_len += length
                if pred == label:
                    entail_correct += 1
                    entail_correct_len += length
                    entail_MRR_c += MRR
                else:
                    entail_MRR_inc += MRR
            elif label == torch.tensor([1]):
                neutral_total += 1
                neutral_total_len += length
                if pred == label:
                    neutral_correct += 1
                    neutral_correct_len += length
                    neutral_MRR_c += MRR
                else:
                    neutral_MRR_inc += MRR
            else:
                contradict_total += 1
                contradict_total_len += length
                if pred == label:
                    contradict_correct += 1
                    contradict_correct_len += length
                    contradict_MRR_c += MRR
                else:
                    contradict_MRR_inc += MRR
    
    return {
        'total':total,
        'total_MRR':round((entail_MRR_c+entail_MRR_inc+
                           neutral_MRR_c+neutral_MRR_inc+
                           contradict_MRR_c+contradict_MRR_inc)/total, 4),
        'total_acc':round((entail_correct+neutral_correct+contradict_correct)/total, 2),
        'total_mean_len':round((entail_total_len+neutral_total_len+contradict_total_len)/total, 1),
        'entail_total':entail_total,
        'entail_acc':round(entail_correct/entail_total, 2),
        'entail_mean_len':round(entail_total_len/entail_total, 1),
        'entail_MRR':round((entail_MRR_c+entail_MRR_inc)/entail_total, 4),
        'entail_correct':entail_correct,
        'entail_correct_mean_len':round(entail_correct_len/entail_correct, 1),
        'entail_MRR_c':round(entail_MRR_c/entail_correct, 4),
        'entail_incorrect':entail_total-entail_correct,
        'entail_incorrect_mean_len':round((entail_total_len-entail_correct_len)/(entail_total-entail_correct), 2),
        'entail_MRR_inc':round(entail_MRR_inc/(entail_total-entail_correct), 4),
        'neutral_total':neutral_total,
        'neutral_acc':round(neutral_correct/neutral_total, 2),
        'neutral_mean_len':round(neutral_total_len/neutral_total, 1),
        'neutral_MRR':round((neutral_MRR_c+neutral_MRR_inc)/neutral_total, 4),
        'neutral_correct':neutral_correct,
        'neutral_correct_mean_len':round(neutral_correct_len/neutral_correct, 1),
        'neutral_MRR_c':round(neutral_MRR_c/neutral_correct, 4),
        'neutral_incorrect':neutral_total-neutral_correct,
        'neutral_incorrect_mean_len':round((neutral_total_len-neutral_correct_len)/(neutral_total-neutral_correct), 2),
        'neutral_MRR_inc':round(neutral_MRR_inc/(neutral_total-neutral_correct), 4),
        'contradict_total':contradict_total,
        'contradict_acc':round(contradict_correct/contradict_total, 2),
        'contradict_mean_len':round(contradict_total_len/contradict_total, 1),
        'contradict_MRR':round((contradict_MRR_c+contradict_MRR_inc)/contradict_total, 4),
        'contradict_correct':contradict_correct,
        'contradict_correct_mean_len':round(contradict_correct_len/contradict_correct, 1),
        'contradict_MRR_c':round(contradict_MRR_c/contradict_correct, 4),
        'contradict_incorrect':contradict_total-contradict_correct,
        'contradict_incorrect_mean_len':round((contradict_total_len-contradict_correct_len)/(contradict_total-contradict_correct), 2),
        'contradict_MRR_inc':round(contradict_MRR_inc/(contradict_total-contradict_correct), 4),
    }
    

def main():
    parser = argparse.ArgumentParser()

    # Required Parameters
    parser.add_argument(
        "--data_dir",
        default="/data",
        type=str,
        required=True,
        help="The input data dir"
    )