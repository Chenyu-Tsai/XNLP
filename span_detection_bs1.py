import argparse
import glob
import logging
import os
import random
import timeit


import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from XLNet import XLNetForMultiSequenceClassification
from multiprocessing import Pool
from functools import partial

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    XLNetTokenizer,
    XLNetConfig,
    get_linear_schedule_with_warmup
)

from processor import SpanDetectionResult, SpanDetectionProcessor, span_detection_convert_example_to_features, Dataset_Span_Detection, SpanDetectionExample
from metrics import span_detection_evaluate
from metrics import compute_predictions_log_probs_bs1 as compute_predictions_log_probs

logger = logging.getLogger(__name__)

MODELS = 'xlnet-base-cased'

MODEL_CLASSES = {
    'xlnet': (XLNetConfig, XLNetForMultiSequenceClassification, XLNetTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    #steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", position=0, leave=True, ncols=100)
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True, ncols=100)
        for step, batch in enumerate(epoch_iterator):
            model.to(args.device)
            model.train()
            input_ids, attention_mask, token_type_ids, start_positions, end_positions, cls_index, p_mask = [t.squeeze(0).to(args.device) for t in batch]
            inputs = {
                      'input_ids':       input_ids,
                      'attention_mask':  attention_mask,
                      'token_type_ids':  token_type_ids,
                      'start_positions': start_positions,
                      'end_positions':   end_positions,
                      'cls_index':       cls_index,
                      'p_mask':          p_mask,
                      'task':                  2,
            }

            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps
            
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        results = evalute(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss
                
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logging.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step

def evalute(args, model, tokenizer, prefix=""): 
    eval_task_names = ("span_detection",)
    eval_dataset = load_and_cache_examples(args, eval_task_names, tokenizer, evaluate=True)

    args.eval_batch_size = args.train_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    device = 'cpu'
    #model.to(device)
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    all_examples = []
    all_features = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True, ncols=100):
        model.eval()
        input_ids, attention_mask, token_type_ids, cls_index, p_mask = [t.squeeze(0).to(args.device) for t in batch[0:5]]

        with torch.no_grad():
            inputs ={
                     "input_ids":      input_ids,
                     "attention_mask": attention_mask,
                     "token_type_ids": token_type_ids,
                     "cls_index":      cls_index,
                     "p_mask":         p_mask,
                     "task":                  2,
            }

            example_index = batch[5]
            unique_id = batch[6]

            outputs = model(**inputs)

            description_text, context_text, span_text, start_position_character = [t[0] for t in batch[-5:-1]]

            example = SpanDetectionExample(
                description_text=description_text,
                context_text=context_text,
                span_text=span_text,
                start_position_character=start_position_character,
                unique_id=unique_id,
            )

            feature = span_detection_convert_example_to_features(
                example,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False,
                example_index=example_index,
                unique_id=unique_id,
            )

            start_logits = outputs[0]
            start_top_index = outputs[1]
            end_logits = outputs[2]
            end_top_index = outputs[3]
            cls_logits = outputs[4]

            result = SpanDetectionResult(
                unique_id,
                start_logits,
                end_logits,
                start_top_index=start_top_index,
                end_top_index=end_top_index,
                cls_logits=cls_logits,
                top_n=model.config.start_n_top,
            )

            all_results.append(result)
            all_examples.append(example)
            all_features.append(feature)
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(eval_dataset))

    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions{}.json".format(prefix))
    output_best_file = os.path.join(args.output_dir, "best_predictions{}.json".format(prefix))

    start_n_top = model.config.start_n_top
    end_n_top = model.config.end_n_top

    predictions = compute_predictions_log_probs(
        all_examples,
        all_features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.min_answer_length,
        output_prediction_file,
        output_nbest_file,
        start_n_top,
        end_n_top,
        tokenizer,
        args.verbose_logging,
    )

    results = span_detection_evaluate(all_examples, predictions, output_best_file)
    return results
    
def load_and_cache_examples(args, task, tokenizer, evaluate=False):

    if not evaluate:
        dataset = Dataset_Span_Detection("rte5_train_span_detection", tokenizer=tokenizer)
    else:
        dataset = Dataset_Span_Detection("rte5_test_span_detection", tokenizer=tokenizer)

    print(type(dataset[0]))
    
    return dataset

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir, Should contain the .tsv file (or other data files) for the task.",
    )

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ",  ".join(MODELS),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the moel predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Question longer than this will be truncated this lenght.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=50,
        type=int,
        help="The maximum length of an answer that can be generated."
    )
    parser.add_argument(
        "--min_answer_length",
        default=5,
        type=int,
        help="Avoiding the span is too short."
    )
    parser.add_argument(
        "verbose_logging",
        action="store_false",
        help="If true, all of the warnings related to data processing will be printed."
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.")
    parser.add_argument("--train_batch_size", default=1, type=int, help="Batch size for training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint evert X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending with step number.",
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cache training and evaluation sets.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization.")
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):

        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir)
        )

    
    # Set CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.warning(
        "Device: %s",
        device,
    )

    # Set seed
    set_seed(args)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        "xlnet-base-cased",
        finetuning_task="span_detection",
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        "xlnet-base-cased",
        do_lower_case=False,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        "xlnet-base-cased",
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:

        train_dataset = load_and_cache_examples(args, "span_detection", tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from-pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Sava a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("trainsformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoint: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("\\")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evalute(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
    return results
            
if __name__ == "__main__":
    print(main())


#python span_detection_bs1.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 256 --evaluate_during_training --learning_rate 2e-5 --num_train_epochs 300.0 --output_dir model_output --overwrite_output_dir --logging_steps 50 --save_steps 600