import logging
from typing import Dict

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

import ujson

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("imdb_review_language_modeling_reader")
class IMDBReviewLanguageModelingReader(DatasetReader):
    """
    Reads the 100K IMDB dataset in format that it appears in
    http://ai.stanford.edu/~amaas/data/sentiment/
    (i.e. this reader expects a full-path to the directory as a result of
     extracting the tar).

    The paper uses strict partitions instead of a sliding window when evaluating TopicRNN as a
    language model to allow fair comparison against other LMs. The variational distribution will
    then only receive the previous BPTT-limit batch of words when recomputing the Gaussian parameters.

    This dataset reader should not be used for training; it should only be used for evaluation.

    Each ``read`` yields a data instance of
        text: A backpropagation-through-time length portion of the review text as a ``TextField``
        stopless_word_frequencies: A ``torch.FloatTensor`` representing the normalized frequencies
            of words in the stopless vocabulary space.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split text into English tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define English token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer(namespace="en", lowercase_tokens=True)}``.
    words_per_instance : ``int``, optional
        The number of words in which the raw text will be bucketed during evaluation.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 words_per_instance: int = 35
                ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(
            start_tokens=[START_SYMBOL],
            end_tokens=[END_SYMBOL]
        )
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="tokens", lowercase_tokens=True)
        }

        self._words_per_instance = words_per_instance

    @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)
        # Break up the text into a series of BPTT chunks and yield one at a time.
        #
        # Strict partitioning instead of a sliding window will mean each chunk is
        # distinct and doesn't not overlap with immediately surrounding chunks.
        with open(cached_path(file_path), 'r') as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                example = ujson.loads(line)
                example_text = example['text']

                example_text_tokenized = self._tokenizer.tokenize(example_text)
                target_text_tokenized = example_text_tokenized[1:]

                tokenized_inputs = []
                tokenized_outputs = []
                for index in range(0, len(example_text_tokenized), self._words_per_instance):
                    tokenized_inputs.append(example_text_tokenized[index:(index + self._words_per_instance)])

                for index in range(1, len(example_text_tokenized), self._words_per_instance):
                    tokenized_outputs.append(target_text_tokenized[index:(index + self._words_per_instance)])

                input_output_pairs = zip(tokenized_inputs, tokenized_outputs)

                previous_batch = None
                for i, (tokenized_input, tokenized_output) in enumerate(input_output_pairs):
                    input_field = TextField(tokenized_input, self._token_indexers)
                    output_field = TextField(tokenized_output, self._token_indexers)
                    example = {
                        'input_tokens': input_field,
                        'output_tokens': output_field,
                        'frequency_tokens': output_field.empty_field()
                    }

                    if i > 0 and previous_batch is not None:
                        example['frequency_tokens'] = previous_batch

                    # When computing perplexity, the topic vector will be drawn from the distrubtion
                    # resulting from this context.
                    previous_batch = input_field

                    yield Instance(example)

    @classmethod
    def from_params(cls, params: Params) -> 'IMDBLanguageModelingReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        words_per_instance = params.pop('words_per_instance', None)
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy,
                   tokenizer=tokenizer,
                   token_indexers=token_indexers,
                   words_per_instance=words_per_instance)


@DatasetReader.register("imdb_review_reader")
class IMDBReviewReader(DatasetReader):
    """
    Reads the 100K IMDB dataset in format that it appears in
    http://ai.stanford.edu/~amaas/data/sentiment/
    (i.e. this reader expects a full-path to the directory as a result of
     extracting the tar).

    This dataset reader will ensure the entire review is available to the model so that the
    variational distribution is as accurate as possible. Unlike the above, training
    is the goal.

    Each ``read`` yields a data instance of
        text: A backpropagation-through-time length portion of the review text as a ``TextField``
        stopless_word_frequencies: A ``torch.FloatTensor`` representing the normalized frequencies
            of words in the stopless vocabulary space.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split text into English tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define English token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer(namespace="en", lowercase_tokens=True)}``.
    words_per_instance : ``int``, optional
        The number of words in which the raw text will be bucketed to allow for more efficient
        training (backpropagation-through-time limit).
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 words_per_instance: int = 35
                ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(
            start_tokens=[START_SYMBOL],
            end_tokens=[END_SYMBOL]
        )
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="tokens", lowercase_tokens=True)
        }

        self._words_per_instance = words_per_instance

    @overrides
    def _read(self, file_path):
        # A training instance consists of the word frequencies for the entire review and a
        # `words_per_instance`` portion of the review.
        file_path = cached_path(file_path)

        # Partition each review into BPTT Limit + 1 chunks to allow room for input (chunk[:-1])
        # and output (chunk[1:]).
        # Break up the text into a series of BPTT chunks and yield one at a time.
        num_tokens = self._words_per_instance + 1
        with open(cached_path(file_path), 'r') as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                example = ujson.loads(line)
                example_text = example['text']
                example_text_tokenized = self._tokenizer.tokenize(example_text)

                # Each review will receive the entire encoded review.
                frequency_field = TextField(example_text_tokenized, self._token_indexers)
                tokenized_strings = []
                for index in range(0, len(example_text_tokenized) - num_tokens, num_tokens - 1):
                    tokenized_strings.append(example_text_tokenized[index:(index + num_tokens)])

                for tokenized_string in tokenized_strings:
                    input_field = TextField(tokenized_string[:-1], self._token_indexers)
                    output_field = TextField(tokenized_string[1:], self._token_indexers)
                    yield Instance({'input_tokens': input_field,
                                    'output_tokens': output_field,
                                    'frequency_tokens': frequency_field})

    @classmethod
    def from_params(cls, params: Params) -> 'IMDBReviewReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        words_per_instance = params.pop('words_per_instance', None)
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy,
                   tokenizer=tokenizer,
                   token_indexers=token_indexers,
                   words_per_instance=words_per_instance)
