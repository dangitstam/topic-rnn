import argparse
import logging
import os
import shutil
from typing import Iterable, List

import torch
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
from library.models.util import description_from_metrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

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
    parser.add_argument("--hidden-size", type=int, default=300,
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
    parser.add_argument("--cuda-device", type=int, default=-1,
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
        BasicTextFieldEmbedder({"tokens": Embedding(train_vocab.get_vocab_size('tokens'),
                                                    args.embedding_dim,
                                                    padding_index=0)}),
        PytorchSeq2SeqWrapper(RNNs[args.rnn_type](args.embedding_dim, args.hidden_size, batch_first=True)),
        topic_dim=300
    )

    if args.cuda_device >= 0:
        model = model.to(torch.device(args.cuda_device))

    # Optimize unfrozen weights only.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.num_epochs):  # Loop over epochs.
        train_epoch(model, train_vocab, train_dataset, validation_dataset, optimizer, args.save_dir, epoch,
                    cuda_device=args.cuda_device)

def train_epoch(model: TopicRNN,
                vocab: Vocabulary,
                train_dataset: Iterable[Instance],
                validation_dataset: Iterable[Instance],
                optimizer: torch.optim.Optimizer,
                serialization_dir: str,
                epoch: int,
                batch_size: int = 32,
                bptt_limit: int = 35,
                cuda_device: int = -1):
    model.train()
    best_model_metrics = None

    # Batch by similar lengths for efficient training.
    iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("input_tokens", "num_tokens")])
    iterator.index_with(vocab)  # Index with the collected vocab.
    train_generator = iterator(train_dataset, num_epochs=1, cuda_device=cuda_device)
    for batch_iteration, batch in enumerate(train_generator):

        # Shape(s): (batch, max_sequence_length), (batch,)
        input_tokens = batch['input_tokens']

        hidden_state = None
        max_sequence_length = input_tokens['tokens'].size(-1)
        bptt_index_generator = tqdm(range(max_sequence_length - bptt_limit))
        for i in bptt_index_generator:
            current_tokens = {'tokens': input_tokens['tokens'][:, i: i + bptt_limit]}
            target_tokens = {'tokens': input_tokens['tokens'][:, (i + 1): (i + 1) + bptt_limit]}

            output_dict, hidden_state = model.forward(current_tokens,
                                                      input_tokens,
                                                      target_tokens=target_tokens,
                                                      hidden_state=hidden_state)

            # Display progress.
            metrics = model.get_metrics()
            description = description_from_metrics(metrics, batch_iteration=batch_iteration)
            bptt_index_generator.set_description(description, refresh=False)

            # Compute gradients and step.
            optimizer.zero_grad()
            output_dict['loss'].backward()
            optimizer.step()

            # Detach hidden state.
            hidden_state = hidden_state.detach()

    # Save the model at the end of each epoch with final validation results, preserving
    # the best seen model the same way AllenNLP does.
    validation_metrics = evaluate(model, vocab, validation_dataset, cuda_device=cuda_device)
    is_best = best_model_metrics is None or validation_metrics['loss'] < best_model_metrics['loss']  # pylint: disable=E1136
    save_checkpoint(model, optimizer, validation_metrics, epoch, serialization_dir, is_best)
    best_model_metrics = validation_metrics

def evaluate(model: TopicRNN,
             vocab: Vocabulary,
             evaluation_dataset: Iterable[Instance],
             batch_size: int = 32,
             bptt_limit: int = 35,
             cuda_device: int = -1):
    model.eval()

    # Batch by similar lengths for efficient training.
    evaluation_iterator = BasicIterator(batch_size=batch_size)
    evaluation_iterator.index_with(vocab)
    evaluation_generator = tqdm(evaluation_iterator(evaluation_dataset,
                                                    num_epochs=1,
                                                    shuffle=False,
                                                    cuda_device=cuda_device,
                                                    for_training=False))

    # Reset metrics and compute them over validation.
    model.get_metrics(reset=True)
    for batch in evaluation_generator:

        # Shape(s): (batch, max_sequence_length), (batch,)
        input_tokens = batch['input_tokens']

        hidden_state = None
        max_sequence_length = input_tokens['tokens'].size(-1)
        bptt_index_generator = tqdm(range(max_sequence_length - bptt_limit))
        for i in bptt_index_generator:
            current_tokens = {'tokens': input_tokens['tokens'][:, i: i + bptt_limit]}
            target_tokens = {'tokens': input_tokens['tokens'][:, (i + 1): (i + 1) + bptt_limit]}
            model.forward(current_tokens, input_tokens, target_tokens=target_tokens, hidden_state=hidden_state)
            metrics = model.get_metrics(reset=True)
            description = description_from_metrics(metrics)
            evaluation_generator.set_description(description)

    # Collect metrics and reset again before resuming training.
    metrics = model.get_metrics(reset=True)

    return metrics

def save_checkpoint(model: TopicRNN,
                    optimizer: torch.optim.Optimizer,
                    validation_metrics: List[float],
                    epoch: int,
                    serialization_dir: str,
                    is_best: bool):
    model_path = os.path.join(serialization_dir, "model_state_epoch_{}.th".format(epoch))
    model_state = model.state_dict()
    torch.save(model_state, model_path)
    if is_best:
        logger.info("Best validation performance so far. "
                    "Copying weights to '%s/best.th'.", serialization_dir)
        shutil.copyfile(model_path, os.path.join(serialization_dir, "best.th"))

    training_state = {'epoch': epoch, 'validation_metrics': validation_metrics,
                      'optimizer': optimizer.state_dict()}
    training_path = os.path.join(serialization_dir, "training_state_epoch_{}.th".format(epoch))
    torch.save(training_state, training_path)


if __name__ == '__main__':
    main()
