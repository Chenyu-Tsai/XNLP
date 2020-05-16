import csv
import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Union
from transformers.tokenization_utils import PreTrainedTokenizer

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

    def get_dev_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "rte5_dev.tsv"))
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

    def get_train_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "rte5_train_multi_label.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % ("train", i)
            text_a = line[0]
            text_b = line[1]
            label = line[8]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(uid=uid, text_a=text_a, text_b=text_b, label=label))
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
                label = line[9]
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(uid=uid, text_a=text_a, text_b=text_b, label=label))
        examples = examples[:pad_len]
        return examples

class SnliProcessor(DataProcessor):
    """ DataProcessor for the SNLI dataset. """

    def get_train_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "snli_train.tsv"))
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
            start_position_character = line[3]
            examples.append(SpanDetectionExample(context_text=context_text, 
                                                 description_text=description_text, 
                                                 span_text=span_text, 
                                                 start_position_character=start_position_character, 
                                                 unique_id=unique_id,))
        return examples

    def get_dev_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "rte5_dev_span_detection.tsv"))
        examples = []
        uid = 100000  
        for i, line in enumerate(lines):
            if i == 0:
                continue
            unique_id = uid + i
            context_text = line[0]
            description_text = line[1]
            span_text = line[2]
            start_position_character = line[3]
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
            start_position_character = line[3]
            examples.append(SpanDetectionExample(context_text=context_text, 
                                                 description_text=description_text, 
                                                 span_text=span_text, 
                                                 start_position_character=start_position_character, 
                                                 unique_id=unique_id,))
        return examples