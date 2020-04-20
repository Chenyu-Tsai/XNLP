import argparse
from utils import *
from transformers import XLNetTokenizer, XLNetPreTrainedModel

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