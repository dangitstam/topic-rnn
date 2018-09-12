import logging

from library.dataset_readers.imdb_review_reader import IMDBReviewReader
from allennlp.data import Vocabulary

logger = logging.getLogger(__name__)


def read_data(imdb_train_path: str,
              imdb_dev_path: str,
              max_vocab_size: int = 5000):
    """
    Read IMDB data into instances while establishing a vocabulary.
    """

    imdb_reader = IMDBReviewReader()
    train_dataset = imdb_reader.read(imdb_train_path)
    logger.info("Read %s training examples", len(train_dataset))

    # Make a vocabulary object from the train set
    train_vocab = Vocabulary.from_instances(train_dataset, max_vocab_size=max_vocab_size)

    # Pre-caution: Unsupervised output won't include positive sentiment.
    train_vocab.add_token_to_namespace("positive", "labels")

    # Read IMDB validation set
    logger.info("Reading IMDB validation set at %s", imdb_dev_path)
    validation_dataset = imdb_reader.read(imdb_dev_path)
    logger.info("Read %s validation examples", len(validation_dataset))

    return train_dataset, train_vocab, validation_dataset
