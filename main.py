import argparse
import logging
import os
import datetime
from data_loader import load_and_cache_examples
from trainer import Trainer
from utils import MODEL_CLASSES, MODEL_PATH_MAP, init_logger, load_tokenizer, set_seed

##### LOGGING CONFIGURATION ####

# Create a directory for log files if it doesn't exist
log_dir = "runtime_logs"
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log filename using timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(log_dir, f"run_{timestamp}.log")

# Create a logger
tcm_logger = logging.getLogger("tcm_logger")
tcm_logger.setLevel(logging.DEBUG)  # Capture all logs

# Create a file handler for logging
file_handler = logging.FileHandler(log_filename, mode="w")
file_handler.setLevel(logging.DEBUG)  # Capture all log levels

# Create a stream handler for logging to the terminal
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)  # Capture all log levels

# Define log format
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to the logger
tcm_logger.addHandler(file_handler)
tcm_logger.addHandler(stream_handler)

# Prevent log propagation to the root logger
tcm_logger.propagate = False

#hf_logger = hf_logging.get_logger("transformers")
#hf_logger.setLevel(logging.INFO)  # Ensure it logs debug messages
#
## Attach the same file handler to Hugging Face's logger
#hf_logger.addHandler(file_handler)
#hf_logger.propagate = False  # Prevent duplicate logs

##### DONE LOGGING CONFIGURATION ####

def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")
    if args.do_eval_dev:
        trainer.load_model()
        trainer.evaluate("dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./PhoATIS", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

    parser.add_argument(
        "--model_type",
        default="phobert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument("--tuning_metric", default="loss", type=str, help="Metrics to tune when training")
    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_eval_dev", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument(
        "--ignore_index",
        default=0,
        type=int,
        help="Specifies a target value that is ignored and does not contribute to the input gradient",
    )

    parser.add_argument("--intent_loss_coef", type=float, default=0.5, help="Coefficient for the intent loss.")
    parser.add_argument(
        "--token_level",
        type=str,
        default="word-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Number of unincreased validation step to wait for early stopping",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="Select gpu id")
    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    # init pretrained
    parser.add_argument("--pretrained", action="store_true", help="Whether to init model from pretrained base model")
    parser.add_argument("--pretrained_path", default="./viatis_xlmr_crf", type=str, help="The pretrained model path")

    # Slot-intent interaction
    parser.add_argument(
        "--use_intent_context_concat",
        action="store_true",
        help="Whether to feed context information of intent into slots vectors (simple concatenation)",
    )
    parser.add_argument(
        "--use_intent_context_attention",
        action="store_true",
        help="Whether to feed context information of intent into slots vectors (dot product attention)",
    )
    parser.add_argument(
        "--attention_embedding_size", type=int, default=200, help="hidden size of attention output vector"
    )

    parser.add_argument(
        "--slot_pad_label",
        default="PAD",
        type=str,
        help="Pad token for slot label pad (to be ignore when calculate loss)",
    )
    parser.add_argument(
        "--embedding_type", default="soft", type=str, help="Embedding type for intent vector (hard/soft)"
    )
    parser.add_argument("--use_attention_mask", action="store_true", help="Whether to use attention mask")

    args = parser.parse_args()
    tcm_logger.info(f"@tcm: In main: use_intent_context_concat: {args.use_intent_context_concat}")
    tcm_logger.info(f"@tcm: In main: use_intent_context_attention: {args.use_intent_context_attention}")

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
