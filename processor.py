import csv
import dataclasses
import json
import logging
import os
import numpy as np
import torch
from functools import partial
from dataclasses import dataclass
from typing import List, Optional, Union
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import XLNetTokenizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from torch.utils.data import TensorDataset, Dataset
import pandas as pd

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
logger = logging.getLogger(__name__)

@dataclass
class InputExample:
    
    uid: str
    text_a: str
    text_b: Optional[str] = None
    text_eval: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """ Serialzes this instance to a Json string. """
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

@dataclass(frozen=True)
class InputFeatures:

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    input_ids_2: Optional[List[int]] = None
    attention_mask_2: Optional[List[int]] = None
    token_type_ids_2: Optional[List[int]] = None
    label_2: Optional[Union[int, float]] = None

    def to_json_string(self):
        """ Serialzes this instance to a Json string. """
        return json.dumps(dataclasses.asdict(self)) + "\n"

class DataProcessor:
    """ Base class for data converters for sequence classification data sets. """

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """ Read a tab separated file """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

class Rte3wayProcessor(DataProcessor):
    """ DataProcessor for the RTE-5 3way classification task. """

    def get_train_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "rte5_train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % ("train", i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(uid=uid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "rte5_test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % ("test", i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(uid=uid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        return ["entailment", "neutral", "contradiction"]

class RteMultiLabelProcessor(DataProcessor):
    """ DataProcessor for the RTE-5 multi-label, we only use the trainset for multi-task learning here. """

    def get_train_examples(self, data_dir, pad_len=None):
        lines = self._read_tsv(os.path.join(data_dir, "rte5_train_multi_label.tsv"))
        examples = []
        if pad_len is not None:
            while len(examples) < pad_len:
                for (i, line) in enumerate(lines):
                    if i == 0:
                        continue
                    uid = "%s-%s" % ("train", i)
                    text_a = line[0]
                    text_b = line[1]
                    label = line[8]
                    assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                    examples.append(InputExample(uid=uid, text_a=text_a, text_b=text_b, label=label))
            examples = examples[:pad_len]
        return examples

    def get_train_examples_span(self, data_dir, pad_len):
        lines = self._read_tsv(os.path.join(data_dir, "rte5_train_multi_label_with_span.tsv"))
        examples = []
        while len(examples) < pad_len:
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                uid = "%s-%s" % ("train", i)
                text_a = line[2]
                text_b = line[1]
                label = line[10]
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(uid=uid, text_a=text_a, text_b=text_b, label=label))
        examples = examples[:pad_len]
        return examples

class SnliProcessor(DataProcessor):
    """ DataProcessor for the SNLI dataset. """

    def get_train_examples(self, data_dir, sample_size=None):
        lines = self._read_tsv(os.path.join(data_dir, "snli_train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if sample_size is not None and i > sample_size:
                break
            uid = "%s-%s" % ("train", i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(uid=uid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "snli_dev.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % ("dev", i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(uid=uid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "snli_test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % ("test", i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(uid=uid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        return ["entailment", "neutral", "contradiction"]

def convert_examples_to_features(
    examples: Union[List[InputExample]],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    labels = [exmaple.label for exmaple in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:3]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.uid))
        logger.info("features: %s" % features[i])
    
    return features

def snli_multi_task_convert_examples_to_features(
    examples_1: Union[List[InputExample]],
    examples_2: Union[List[InputExample]],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    labels_1 = [exmaple.label for exmaple in examples_1]
    labels_2 = [exmaple.label for exmaple in examples_2]

    batch_encoding_1 = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples_1], max_length=max_length, pad_to_max_length=True,
    )
    batch_encoding_2 = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples_2], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples_1)):
        inputs_1 = {k: batch_encoding_1[k][i] for k in batch_encoding_1}
        inputs_2 = {k: batch_encoding_2[k][i] for k in batch_encoding_2}
        feature = InputFeatures(input_ids=inputs_1['input_ids'],
                                attention_mask=inputs_1['attention_mask'],
                                token_type_ids=inputs_1['token_type_ids'],
                                label=labels_1[i],
                                input_ids_2=inputs_2['input_ids'],
                                attention_mask_2=inputs_2['attention_mask'],
                                token_type_ids_2=inputs_2['token_type_ids'],
                                label_2=labels_2[i],
        )
        features.append(feature)

    for i, example in enumerate(examples_1[:3]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.uid))
        logger.info("features: %s" % features[i])
    
    return features
class SpanDetectionExample(object):
    """
    A single training/test example for the RTE-5 span detection.
    Args:
        context_text: The context string
        description_text: The description string
        span_text: The span straing
        start_position_character: The character position of the start of the span
        unique_id: The example's unique identifier
        pred: Used during evaluation
    """

    def __init__(self, 
                 context_text, 
                 description_text, 
                 span_text,
                 start_position_character,
                 unique_id=None,
                 pred=None,
                 ):
        self.context_text = context_text
        self.description_text = description_text
        self.span_text = span_text
        self.unique_id = unique_id
        self.pred = pred

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if self._is_whitespace(c):
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
                min(start_position_character + len(span_text) - 1, len(char_to_word_offset) - 1)
            ]

    def _is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

class SpanDetectionFeatures(object):

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

class SpanDetectionResult(object):
    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None, top_n=None, pred=None, label=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        self.start_top_index = start_top_index
        self.end_top_index = end_top_index
        self.cls_logits = cls_logits
        self.top_n = top_n
        self.pred = pred
        self.label = label

class SpanDetectionProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "rte5_train_span_detection.tsv"))
        examples = []
        uid = 100000  
        for i, line in enumerate(lines):
            if i == 0:
                continue
            unique_id = uid + i
            context_text = line[0]
            description_text = line[1]
            span_text = line[2]
            start_position_character = int(line[3])
            examples.append(SpanDetectionExample(context_text=context_text, 
                                                 description_text=description_text, 
                                                 span_text=span_text, 
                                                 start_position_character=start_position_character, 
                                                 unique_id=unique_id,))
        return examples

    def get_test_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "rte5_test_span_detection.tsv"))
        examples = []
        uid = 100000  
        for i, line in enumerate(lines):
            if i == 0:
                continue
            unique_id = uid + i
            context_text = line[0]
            description_text = line[1]
            span_text = line[2]
            start_position_character = int(line[3])
            examples.append(SpanDetectionExample(context_text=context_text, 
                                                 description_text=description_text, 
                                                 span_text=span_text, 
                                                 start_position_character=start_position_character, 
                                                 unique_id=unique_id,))
        return examples

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

def span_detection_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training, example_index=None, unique_id=None):
    features = []

    example_index = example_index.item() if example_index else 0
    unique_id = unique_id.item() if unique_id else 0

    if is_training:
        start_position = example.start_position
        end_position = example.end_position

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.span_text
        )

    spans = []
    
    truncated_query = tokenizer.encode(example.description_text, add_special_tokens=False, max_length=max_query_length)
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
        if is_training:
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SpanDetectionFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=example_index,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=unique_id,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
            )
        )

    return features

def span_detection_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

def span_detection_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):


    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=span_detection_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            span_detection_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert span detection examples to features",
                disable=not tqdm_enabled,
            )
        )
    new_features = []
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            new_features.append(example_feature)
        example_index += 1
    features = new_features
    del new_features
    if return_dataset:
        #Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)


        if not is_training:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
            )

        return features, dataset
    else:
        return features

class Dataset_Span_Detection(Dataset):
    
    def __init__(self, mode, tokenizer):
        assert mode in ["rte5_train_span_detection", "rte5_test_span_detection"]
        self.mode = mode
        self.dir = "data/"
        self.df = pd.read_csv(self.dir + mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        if self.mode == "rte5_test_span_detection":
            context_text, description_text, span_text, start_position_character = self.df.iloc[idx,:4].values
            labels = self.df.iloc[idx, -1]
            labels = torch.tensor(labels)
        else:
            context_text, description_text, span_text, start_position_character, entail_label = self.df.iloc[idx,:].values
        
        example = SpanDetectionExample(
            description_text=description_text,
            context_text=context_text,
            span_text=span_text,
            start_position_character=start_position_character
        )

        features = span_detection_convert_example_to_features(example,
                                                     max_seq_length=384,
                                                     doc_stride=128,
                                                     max_query_length=64,
                                                     is_training= True if self.mode == "rte5_train_span_detection" else False,
                                                    )
        input_ids = torch.tensor(features[0].input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(features[0].attention_mask, dtype=torch.long).unsqueeze(0)
        token_type_ids = torch.tensor(features[0].token_type_ids, dtype=torch.long).unsqueeze(0)
        start_position = torch.tensor(features[0].start_position, dtype=torch.long).unsqueeze(0)
        end_position = torch.tensor(features[0].end_position, dtype=torch.long).unsqueeze(0)
        cls_index = torch.tensor(features[0].cls_index, dtype=torch.long).unsqueeze(0)
        p_mask = torch.tensor(features[0].p_mask, dtype=torch.float).unsqueeze(0)
        example_index = idx
        unique_id = 1000001 + idx
        
        if self.mode == "rte5_test_span_detection":
            return (
                    input_ids, 
                    attention_mask, 
                    token_type_ids, 
                    cls_index, 
                    p_mask, 
                    example_index, 
                    unique_id, 
                    description_text,
                    context_text,
                    span_text,
                    start_position_character,
                    labels,
                    )
        else:
            return (input_ids, attention_mask, token_type_ids, start_position, end_position, cls_index, p_mask)

    def __len__(self):
        return self.len

class three_multi_tasks(Dataset):
    
    def __init__(self, mode, tokenizer, processor):
        assert mode in ["rte5_train_span_detection"]
        self.mode = mode
        self.dir = "data/"
        self.df = pd.read_csv(self.dir + mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer
        self.multi_label_examples = processor.get_train_examples('data/', pad_len=self.len)
        
    def __getitem__(self, idx):
        def _return_multi_label(label):
            label = label.replace('[', '').replace(']','')
            label = np.fromstring(label, dtype=int, sep=',')
            return label

        context_text, description_text, span_text, start_position_character, entail_label = self.df.iloc[idx,:].values
        entail_label = torch.tensor(entail_label, dtype=torch.long).unsqueeze(0)
        m_text_a = self.multi_label_examples[idx].text_a
        m_text_b = self.multi_label_examples[idx].text_b
        m_labels = _return_multi_label(self.multi_label_examples[idx].label)
        m_labels_tensor = torch.tensor(m_labels, dtype=torch.float)
            
        m_inputs = tokenizer.encode_plus(m_text_a, m_text_b, return_tensors='pt', add_special_tokens=True)
        m_tokens_tensor = m_inputs['input_ids']
        m_segments_tensor = m_inputs['token_type_ids']
        m_masks_tensor = m_inputs['attention_mask']
        
        example = SpanDetectionExample(
            description_text=description_text,
            context_text=context_text,
            span_text=span_text,
            start_position_character=start_position_character
        )

        features = span_detection_convert_example_to_features(example,
                                                     max_seq_length=384,
                                                     doc_stride=128,
                                                     max_query_length=64,
                                                     is_training= True,
                                                    )
        input_ids = torch.tensor(features[0].input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(features[0].attention_mask, dtype=torch.long).unsqueeze(0)
        token_type_ids = torch.tensor(features[0].token_type_ids, dtype=torch.long).unsqueeze(0)
        start_position = torch.tensor(features[0].start_position, dtype=torch.long).unsqueeze(0)
        end_position = torch.tensor(features[0].end_position, dtype=torch.long).unsqueeze(0)
        cls_index = torch.tensor(features[0].cls_index, dtype=torch.long).unsqueeze(0)
        p_mask = torch.tensor(features[0].p_mask, dtype=torch.float).unsqueeze(0)
        example_index = idx
        unique_id = 1000001 + idx
        
        return (
            input_ids, 
            attention_mask, 
            token_type_ids, 
            start_position, 
            end_position, 
            cls_index, 
            p_mask, 
            entail_label,
            m_tokens_tensor,
            m_masks_tensor,
            m_segments_tensor,
            m_labels_tensor,
            )

    def __len__(self):
        return self.len