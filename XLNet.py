import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers.modeling_utils import (WEIGHTS_NAME, PretrainedConfig, PreTrainedModel,
                             SequenceSummary, PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits)
from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetPreTrainedModel, XLNetModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from torch.utils.data import TensorDataset



tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class XLNetForMultiSequenceClassification(XLNetPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 3
        # RTE Task
        self.num_labels_3way = 3
        # RTE SPs multi-label task
        self.num_labels_multi = 5
        # RTE span detection task
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj_3way = nn.Linear(config.d_model, self.num_labels_3way)
        self.logits_proj_multi = nn.Linear(config.d_model, self.num_labels_multi)
        self.weights_3way = [1, 1.4, 3.3]
        self.weights_multi = [15, 10, 15, 5, 5]
        self.class_weights_3way = torch.FloatTensor(self.weights_3way).to(device)
        self.class_weights_multi = torch.FloatTensor(self.weights_multi).to(device)

        # RTE span detection task
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)
        
        self.init_weights()
        

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        mems=None, 
        perm_mask=None, 
        target_mapping=None,
        token_type_ids=None,
        input_mask=None, 
        head_mask=None, 
        labels=None, 
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        cls_index=None,
        p_mask=None,
        use_cache=None,
        task=None,
    ):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               mems=mems,
                                               perm_mask=perm_mask,
                                               target_mapping=target_mapping,
                                               token_type_ids=token_type_ids,
                                               input_mask=input_mask, 
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds,
                                              )
    
        hidden_states = transformer_outputs[0]
        if task == 0 or task == 1 or task == None:
            output = self.sequence_summary(hidden_states)
            
            if labels is None:
                logits = self.logits_proj_3way(output)
                outputs = (logits,) + transformer_outputs[1:]

            if labels is not None:
                task_check = 0
            
                if labels.size() == torch.Size([1]):
                    logits_3way = self.logits_proj_3way(output)
                    outputs = (logits_3way,) + transformer_outputs[1:]
                    task_check = 1
                else:
                    logits_multi = self.logits_proj_multi(output)
                    outputs = (logits_multi,) + transformer_outputs[1:]

                if task_check:
                    loss_fct = CrossEntropyLoss(weight=self.class_weights_3way)
                    loss = loss_fct(logits_3way.view(-1, self.num_labels_3way), labels.view(-1)).to(device)
                else:
                    loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights_multi)
                    loss = loss_fct(logits_multi.view(-1, self.num_labels_multi), labels).to(device)
                outputs = (loss,) + outputs
        else:
            start_logits = self.start_logits(hidden_states, p_mask=p_mask)

            outputs = transformer_outputs[1:]

            if start_positions is not None and end_positions is not None:

                end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

                loss_fct = CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions).to(device)
                #print(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions).to(device)
                #print(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                #print(start_loss, end_loss)

                if cls_index is not None:
                    cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                    loss_fct_cls = nn.BCEWithLogitsLoss()
                    cls_loss = loss_fct_cls(cls_logits, torch.tensor([0], dtype=torch.float, device=device)).to(device)
                    total_loss += cls_loss * 0.5
                    #print(cls_loss)

                outputs = (total_loss,) + outputs
            
        return outputs

class Dataset_multi(Dataset):
    
    def __init__(self, mode, tokenizer):
        assert mode in ["data/train_multi_label"]
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.df = self.df[['text_a', 'text_b', 'labels']]
        self.len = len(self.df)
        self.label_map = {'True': 0, 'False': 1}
        self.tokenizer = tokenizer
        self.task = 1

    def __getitem__(self, idx):
        text_a, text_b = self.df.iloc[idx, :2].values
        
        label = self.df.iloc[idx, 2].replace('[', '')
        label = label.replace(']','')
        label = np.fromstring(label, dtype=int, sep=',')
        label_tensor = torch.tensor(label, dtype=torch.float).unsqueeze(0)
            
        inputs = tokenizer.encode_plus(text_a, text_b, return_tensors='pt', add_special_tokens=True)
        tokens_tensor = inputs['input_ids']
        segments_tensor = inputs['token_type_ids']
        masks_tensor = inputs['attention_mask']

        task = self.task

        return (task, tokens_tensor, segments_tensor, masks_tensor, label_tensor)
    
    def __len__(self):
        return self.len

class Dataset_3Way(Dataset):
    
    def __init__(self, mode, tokenizer):
        assert mode in ["data/RTE5_train", "data/RTE5_test"]
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer
        self.task = 0
        
    def __getitem__(self, idx):
        if self.mode == "data/RTE5_test":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
            task = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label).unsqueeze(0)
            task = self.task
            
        inputs = tokenizer.encode_plus(text_a, text_b, return_tensors='pt', add_special_tokens=True)
        tokens_tensor = inputs['input_ids']
        segments_tensor = inputs['token_type_ids']
        masks_tensor = inputs['attention_mask']
            
        return (task, tokens_tensor, segments_tensor, masks_tensor, label_tensor)
    
    def __len__(self):
        return self.len

class Dataset_Span_Detection(Dataset):
    
    def __init__(self, mode, tokenizer):
        assert mode in ["data/train_span_detection"]
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer
        self.task = 2
        
    def __getitem__(self, idx):
        context_text, question_text, answer_text, start_position_character = self.df.iloc[idx,:].values
        
        example = SquadExample(
            question_text=question_text,
            context_text=context_text,
            answer_text=answer_text,
            start_position_character=start_position_character
        )

        features = squad_convert_example_to_features(example,
                                                     max_seq_length=384,
                                                     doc_stride=128,
                                                     max_query_length=64,
                                                    )
        input_ids = torch.tensor(features[0].input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(features[0].attention_mask, dtype=torch.long).unsqueeze(0)
        token_type_ids = torch.tensor(features[0].token_type_ids, dtype=torch.long).unsqueeze(0)
        start_position = torch.tensor(features[0].start_position, dtype=torch.long).unsqueeze(0)
        end_position = torch.tensor(features[0].end_position, dtype=torch.long).unsqueeze(0)
        cls_index = torch.tensor(features[0].cls_index, dtype=torch.long).unsqueeze(0)
        p_mask = torch.tensor(features[0].p_mask, dtype=torch.float).unsqueeze(0)
        task = self.task
        
        return (task, input_ids, attention_mask, token_type_ids, start_position, end_position, cls_index, p_mask)

    def __len__(self):
        return self.len

class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
    """

    def __init__(
        self,
        question_text,
        context_text,
        answer_text,
        start_position_character,
    ):
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

class SquadFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position

def squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length):
    features = []
    start_position = example.start_position
    end_position = example.end_position

        # # If the answer cannot be found in the text, then skip this example.
        # actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        # cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        # if actual_text.find(cleaned_answer_text) == -1:
        #     logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
        #     return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)


    tok_start_position = orig_to_tok_index[example.start_position]
    if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
    else:
        tok_end_position = len(all_doc_tokens) - 1
    (tok_start_position, tok_end_position) = _improve_answer_span(
        all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
    )

    spans = []

    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
    sequence_added_tokens = tokenizer.max_len - tokenizer.max_len_single_sentence
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair
    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):
        encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=False,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict:
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])

        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        start_position = 0
        end_position = 0
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = span["start"]
        doc_end = span["start"] + span["length"] - 1
        out_of_span = False

        if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
            out_of_span = True

        if out_of_span:
            start_position = cls_index
            end_position = cls_index
            span_is_impossible = True
        else:
            if tokenizer.padding_side == "left":
                doc_offset = 0
            else:
                doc_offset = len(truncated_query) + sequence_added_tokens

            start_position = tok_start_position - doc_start + doc_offset
            end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
            )
        )
    return features


def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert