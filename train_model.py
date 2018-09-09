import argparse
import logging
import os
from typing import Iterable

import torch
import torch.nn as nn
from allennlp.data.instance import Instance
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import \
    PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders.basic_text_field_embedder import \
    BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from tqdm import tqdm

from data import read_data
from library.models.topic_rnn import TopicRNN
from library.models.util import description_from_metrics, rnn_forward

logger = logging.getLogger(__name__)

RNNs = {
    "lstm": torch.nn.LSTM,
    "gru": torch.nn.GRU,
    "vanilla": torch.nn.RNN
}

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))

    parser.add_argument("--imdb-train-path", type=str,
                        default=os.path.join(
                            project_root, "data", "valid_unsup.jsonl"),
                        help="Path to the IMDB training data.")
    parser.add_argument("--imdb-dev-path", type=str,
                        default=os.path.join(
                            project_root, "data", "valid_unsup.jsonl"),
                        help="Path to the IMDB dev data.")
    parser.add_argument("--imdb-test-path", type=str,
                        default=os.path.join(
                            project_root, "data", "test.jsonl"),
                        help="Path to the IMDB test data.")
    parser.add_argument("--load-path", type=str,
                        help=("Path to load a saved model from and "
                              "evaluate on test data. May not be "
                              "used with --save-dir."))
    parser.add_argument("--save-dir", type=str,
                        help=("Path to save model checkpoints and logs. "
                              "Required if not using --load-path. "
                              "May not be used with --load-path."))
    parser.add_argument("--rnn-type", type=str, default="vanilla",
                        choices=["lstm", "gru", "vanilla"],
                        help="Model type to train.")
    parser.add_argument("--embedding-dim", type=int, default=100,
                        help="Number of dimensions for word embeddings.")
    parser.add_argument("--max-vocab-size", type=int, default=5000,
                        help=("Maximum number of tokens allowed in the vocabulary."))
    parser.add_argument("--max-passage-length", type=int, default=150,
                        help="Maximum number of words in the passage.")
    parser.add_argument("--max-question-length", type=int, default=15,
                        help="Maximum number of words in the question.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size to use in training and evaluation.")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden size to use in the RNN.")
    parser.add_argument("--num-epochs", type=int, default=25,
                        help="Number of epochs to train for.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout proportion.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="The learning rate to use.")
    parser.add_argument("--log-period", type=int, default=50,
                        help=("Update training metrics every "
                              "log-period weight updates."))
    parser.add_argument("--validation-period", type=int, default=500,
                        help=("Calculate metrics on validation set every "
                              "validation-period weight updates."))
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed to use")
    parser.add_argument("--cuda-device", default=-1,
                        help="Train or evaluate with GPU.")
    parser.add_argument("--demo", action="store_true",
                        help="Run the interactive web demo.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to use for web demo.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to use for web demo.")
    args = parser.parse_args()

    # Read the training and validaton dataset into lists of instances
    # and get a vocabulary from the train set.
    train_dataset, train_vocab, validation_dataset = read_data(
        args.imdb_train_path, args.imdb_dev_path, args.max_vocab_size)

    # Save the train_vocab to a file.
    vocab_dir = os.path.join(args.save_dir, "train_vocab")
    logger.info("Saving train vocabulary to %s", vocab_dir)
    train_vocab.save_to_files(vocab_dir)

    # Define the model.
    model = TopicRNN(
        train_vocab,
        BasicTextFieldEmbedder({"tokens": Embedding(train_vocab.get_vocab_size('tokens'), args.embedding_dim,
            padding_index=0)}),
        PytorchSeq2SeqWrapper(RNNs[args.rnn_type](args.embedding_dim, args.hidden_size, batch_first=True))
    )

    if args.cuda_device >= 0:
        model = model.to(torch.device(args.cuda_device))

    # Optimize unfrozen weights only.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for _ in range(10):  # Loop over epochs.
        train_epoch(model, train_vocab, train_dataset, validation_dataset, optimizer,
                    args.save_dir, cuda_device=args.cuda_device)

def train_epoch(model: TopicRNN,
                vocab: Vocabulary,
                train_dataset: Iterable[Instance],
                validation_dataset: Iterable[Instance],
                optimizer: torch.optim.Optimizer,
                save_dir: str,
                batch_size: int = 32,
                bptt_limit: int = 35,
                cuda_device: int = -1):
    model.train()

    # Batch by similar lengths for efficient training.
    iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("input_tokens", "num_tokens")])
    iterator.index_with(vocab)  # Index with the collected vocab.
    train_generator = iterator(train_dataset, num_epochs=1, cuda_device=cuda_device)
    for batch_iteration, batch in enumerate(train_generator):
        print("Batch {}:".format(batch_iteration))

        # Shape(s): (batch, max_sequence_length), (batch,)
        input_tokens = batch['input_tokens']

        hidden_state = None
        max_sequence_length = input_tokens['tokens'].size(-1)
        bptt_index_generator = tqdm(range(max_sequence_length - bptt_limit))
        for i in bptt_index_generator:
            current_tokens = {'tokens': input_tokens['tokens'][:, i: i + bptt_limit]}
            target_tokens = {'tokens': input_tokens['tokens'][:, (i + 1): (i + 1) + bptt_limit]}

            output_dict, hidden_state = model.forward(current_tokens, target_tokens=target_tokens,
                hidden_state=hidden_state)

            # Display progress.
            metrics = model.get_metrics()
            description = description_from_metrics(metrics)
            bptt_index_generator.set_description(description, refresh=False)

            # Compute gradients and step.
            optimizer.zero_grad()
            output_dict['loss'].backward()
            optimizer.step()

            # Detach hidden state.
            hidden_state = hidden_state.detach()

    # TODO: Make evaluate that uses vocab

if __name__ == '__main__':
    main()
