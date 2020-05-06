import csv
import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

@dataclass
class InputExample:
    
    guid: str
    text_a: str
    text_b: Optional[str] = None
    text_eval: Optional[str] = None
    label: Optional[str] = None
    multi_label: Optional[str] = None

    def to_json_string(self):
        """ Serialzes this instance to a Json string. """
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

@dataclass(frozen=True)
class InputFeatures:

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """ Serialzes this instance to a Json string. """
        return json.dumps(dataclasses.asdict(self)) + "\n"

class DataProcessor:
    """ Base class for data converters for sequence classification data sets. """

    def get_example_from_tensor_dict(self, tensor_dict):
        """ Gets an example from a dict with tensors """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """ Get collection of InputExample's for the train set. """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """ Gets a collection of InputExample's for the dev set. """
        raise NotImplementedError()

    def get_labels(self):
        """ Gets the list of labels for this data set. """
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """ Read a tab separated file """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

