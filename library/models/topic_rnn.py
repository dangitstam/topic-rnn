from collections import Counter
from typing import Dict, Optional

import torch
import torch.nn as nn
from allennlp.data.vocabulary import (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN,
                                      Vocabulary)
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.modules import (FeedForward, Seq2SeqEncoder, TextFieldEmbedder,
                              TimeDistributed)
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import \
    PytorchSeq2VecWrapper
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import Average, CategoricalAccuracy
from overrides import overrides
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import sigmoid
from torch.nn.modules.linear import Linear

from library.dataset_readers.util import STOP_WORDS
from library.metrics.perplexity import Perplexity


@Model.register("topic_rnn")
class TopicRNN(Model):
    """
    Replication of Dieng et al.'s
    ``TopicRnn: A Recurrent Neural Network with Long-range Semantic Dependency``
    (https://arxiv.org/abs/1611.01702).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    text_encoder : ``Seq2SeqEncoder``
        The encoder used to encode input text.
    text_decoder: ``Seq2SeqEncoder``
        Projects latent word representations into probabilities over the vocabulary.
    variational_autoencoder : ``FeedForward``
        The feedforward network to produce the parameters for the variational distribution.
    topic_dim: ``int``
        The number of latent topics to use.
    freeze_feature_extraction: ``bool``, optional
        If true, the encoding of text as well as learned topics will be frozen.
    classification_mode: ``bool``, optional
        If true, the model will output cross entropy loss w.r.t sentiment instead of
        prediction the rest of the sequence.
    pretrained_file: ``str``, optional
        If provided, will initialize the model with the weights provided in this file.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 variational_autoencoder: FeedForward = None,
                 sentiment_classifier: FeedForward = None,
                 topic_dim: int = 20,
                 freeze_feature_extraction: bool = False,
                 classification_mode: bool = False,
                 pretrained_file: str = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TopicRNN, self).__init__(vocab, regularizer)

        self.metrics = {
            'cross_entropy': Average(),
            'mapped_term_freq_sum': Average()
        }

        self.classification_mode = classification_mode
        if classification_mode:
            self.metrics['sentiment'] = CategoricalAccuracy()

        if pretrained_file:
            archive = load_archive(pretrained_file)
            pretrained_model = archive.model
            self._init_from_archive(pretrained_model)
        else:
            # Model parameter definition.
            #
            # Defaults reflect Dieng et al.'s decisions when training their semi-unsupervised
            # IMDB sentiment classifier.
            self.text_field_embedder = text_field_embedder
            self.vocab_size = self.vocab.get_vocab_size("tokens")
            self.text_encoder = text_encoder
            self.topic_dim = topic_dim
            self.vocabulary_projection_layer = TimeDistributed(Linear(text_encoder.get_output_dim(),
                                                                      self.vocab_size))

            # Parameter gamma from the paper; projects hidden states into binary logits for whether a
            # word is a stopword.
            self.stopword_pojection_layer = TimeDistributed(Linear(text_encoder.get_output_dim(), 2))

            self.tokens_to_index = vocab.get_token_to_index_vocabulary()

            # This step should only ever be performed ONCE.
            # When running allennlp train, the vocabulary will be constructed before the model instantiation, but
            # we can't create the stopless namespace until we get here.
            # Check if there already exists a stopless namespace: if so refrain from altering it.
            if "stopless" not in vocab._token_to_index.keys():
                assert self.tokens_to_index[DEFAULT_PADDING_TOKEN] == 0 and \
                       self.tokens_to_index[DEFAULT_OOV_TOKEN] == 1
                for token, _ in self.tokens_to_index.items():
                    if token not in STOP_WORDS:
                        vocab.add_token_to_namespace(token, "stopless")

                # Since a vocabulary with the stopless namespace hasn't been saved, save one for convienience.
                vocab.save_to_files("vocabulary")

            # Compute stop indices in the normal vocab space to prevent stop words
            # from contributing to the topic additions.
            self.stop_indices = torch.LongTensor([vocab.get_token_index(stop) for stop in STOP_WORDS])

            # Learnable topics.
            # TODO: How should these be initialized?
            self.beta = nn.Parameter(torch.rand(topic_dim, self.vocab_size))

            # mu: The mean of the variational distribution.
            self.mu_linear = nn.Linear(topic_dim, topic_dim)

            # sigma: The root standard deviation of the variational distribution.
            self.sigma_linear = nn.Linear(topic_dim, topic_dim)

            # noise: used when sampling.
            self.noise = MultivariateNormal(torch.zeros(topic_dim), torch.eye(topic_dim))

            stopless_dim = vocab.get_vocab_size("stopless")
            self.variational_autoencoder = variational_autoencoder or FeedForward(
                # Takes as input the word frequencies in the stopless dimension and projects
                # the word frequencies into a latent topic representation.
                #
                # Each latent representation will help tune the variational dist.'s parameters.
                stopless_dim,
                3,
                [500, 500, topic_dim],
                torch.nn.ReLU(),
            )

            # The shape for the feature vector for sentiment classification.
            # (RNN Hidden Size + Inference Network output dimension).
            sentiment_input_size = text_encoder.get_output_dim() + topic_dim
            self.sentiment_classifier = sentiment_classifier or FeedForward(
                # As done by the paper; a simple single layer with 50 hidden units
                # and sigmoid activation for sentiment classification.
                sentiment_input_size,
                2,
                [50, 2],
                torch.nn.Sigmoid(),
            )

        if freeze_feature_extraction:
            # Freeze the RNN and VAE pipeline so that only the classifier is trained.
            for name, param in self.named_parameters():
                if "sentiment_classifier" not in name:
                    param.requires_grad = False

        self.sentiment_criterion = nn.CrossEntropyLoss()

        self.num_samples = 10

        initializer(self)

    def _init_from_archive(self, pretrained_model: Model):
        """ Given a TopicRNN instance, take its weights. """
        self.text_field_embedder = pretrained_model.text_field_embedder
        self.vocab_size = pretrained_model.vocab_size
        self.text_encoder = pretrained_model.text_encoder

        # This function is only to be invoved when needing to classify.
        # To avoid manually dealing with padding, instantiate a Seq2Vec instead.
        self.text_to_vec = PytorchSeq2VecWrapper(self.text_encoder._modules['_module'])

        self.topic_dim = pretrained_model.topic_dim
        self.vocabulary_projection_layer = pretrained_model.vocabulary_projection_layer
        self.tokens_to_index = pretrained_model.tokens_to_index
        self.stop_indices = pretrained_model.stop_indices
        self.beta = pretrained_model.beta
        self.mu_linear = pretrained_model.mu_linear
        self.sigma_linear = pretrained_model.sigma_linear
        self.noise = pretrained_model.noise
        self.variational_autoencoder = pretrained_model.variational_autoencoder
        self.sentiment_classifier = pretrained_model.sentiment_classifier

    @overrides
    def forward(self,  # type: ignore
                input_tokens: Dict[str, torch.LongTensor],
                output_tokens: Dict[str, torch.LongTensor],
                frequency_tokens: Dict[str, torch.LongTensor],
                sentiment: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_tokens : Dict[str, Variable], required
            The BPTT portion of text to encode.
        output_tokens : Dict[str, Variable], required
            The BPTT portion of text to produce.
        word_counts : Dict[str, int], required
            Words mapped to the frequency in which they occur in their source text.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        output_dict = {}

        # Encode the input text.
        # Shape: (batch x sequence length x hidden size)
        embedded_input = self.text_field_embedder(input_tokens)
        input_mask = util.get_text_field_mask(input_tokens)
        encoded_input = self.text_encoder(embedded_input, input_mask)

        # Initial projection into vocabulary space, v^T * h_t.
        # Shape: (batch x sequence length x vocabulary size)
        logits = self.vocabulary_projection_layer(encoded_input)

        # Word frequency vectors and noise aren't generated with the model. If the model
        # is running on a GPU, these tensors need to be moved to the correct device.
        device = logits.device

        # Mask the output for proper loss calculation.
        output_mask = util.get_text_field_mask(output_tokens)
        relevant_output = output_tokens['tokens'].contiguous()
        relevant_output_mask = output_mask.contiguous()

        # Compute Gaussian parameters.
        stopless_word_frequencies = self._compute_word_frequency_vector(frequency_tokens).to(device=device)
        mapped_term_frequencies = self.variational_autoencoder(stopless_word_frequencies)

        # If the inference network ever learns to output just 0, something has gone wrong.
        assert mapped_term_frequencies.sum().item() > 0

        mu = self.mu_linear(mapped_term_frequencies)
        log_sigma = self.sigma_linear(mapped_term_frequencies)

        aggregate_theta_probability = 0
        aggregate_cross_entropy_loss = 0
        for _ in range(self.num_samples):

            # Compute noise for sampling.
            epsilon = self.noise.rsample().to(device=device)

            # Compute noisy topic proportions given Gaussian parameters.
            theta = mu + torch.exp(log_sigma) * epsilon

            # I. Compute the integral of q(theta) log P(theta)
            aggregate_theta_probability += theta ** 2

            # II. Compute cross entropy against next words for the current sample of noise.
            # Padding and OOV tokens are indexed at 0 and 1.
            topic_additions = torch.mm(theta, self.beta)
            topic_additions.t()[self.stop_indices] = 0  # Stop words have no contribution via topics.
            topic_additions.t()[0] = 0                  # Padding will be treated as stops.
            topic_additions.t()[1] = 0                  # Unknowns will be treated as stops.
            topic_additions = topic_additions.unsqueeze(1).expand_as(logits)
            cross_entropy_loss = util.sequence_cross_entropy_with_logits(logits + topic_additions,
                                                                         relevant_output,
                                                                         relevant_output_mask)
            aggregate_cross_entropy_loss += cross_entropy_loss

        averaged_theta_probability = -aggregate_theta_probability.sum() / (2 * self.num_samples)
        averaged_cross_entropy_loss = aggregate_cross_entropy_loss / self.num_samples

        # III. Compute stopword probabilities and gear RNN hidden states toward learning them. 
        stopword_logits = sigmoid(self.stopword_pojection_layer(encoded_input))
        relevant_stopword_output = self._compute_stopword_mask(output_tokens).contiguous().to(device=device)
        stopword_loss = util.sequence_cross_entropy_with_logits(stopword_logits,
                                                                relevant_stopword_output,
                                                                relevant_output_mask)

        # IV. Compute the integral q(theta) log q(theta).
        log_sigma_sum = log_sigma.sum()

        if self.classification_mode:
            output_dict['loss'] = self._classify_sentiment(frequency_tokens, mapped_term_frequencies, sentiment)
        else:
            # Negate everything but the cross entropy loss since it is already defined as negative log likelihood.
            output_dict['loss'] = -averaged_theta_probability + averaged_cross_entropy_loss - stopword_loss - log_sigma_sum

        # It's nice to see how the model does as a language model.
        self.metrics['cross_entropy'](averaged_cross_entropy_loss.item())

        return output_dict

    def _classify_sentiment(self,  # type: ignore
                            frequency_tokens: Dict[str, torch.LongTensor],
                            mapped_term_frequencies: torch.Tensor,
                            sentiment: Dict[str, torch.LongTensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ
        """
        Using the entire review (frequency_tokens), classify it as positive or negative.
        """

        # Encode the input text.
        # Shape: (batch, sequence length, hidden size)
        embedded_input = self.text_field_embedder(frequency_tokens)
        input_mask = util.get_text_field_mask(frequency_tokens)

        # Use text_to_vec to avoid dealing with padding.
        encoded_input = self.text_to_vec(embedded_input, input_mask)

        # Construct feature vector.
        # Shape: (batch, RNN hidden size + number of topics)
        sentiment_features = torch.cat([encoded_input, mapped_term_frequencies], dim=-1)

        # Classify.
        logits = self.sentiment_classifier(sentiment_features)
        loss = self.sentiment_criterion(logits, sentiment)

        self.metrics['sentiment'](logits, sentiment)

        return loss

    def _compute_word_frequency_vector(self, frequency_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """ Given the window in which we're allowed to collect word frequencies, produce a
            vector in the 'stopless' dimension for the variational distribution.
        """
        batch_size = frequency_tokens['tokens'].size(0)
        res = torch.zeros(batch_size, self.vocab.get_vocab_size("stopless"))
        for i, row in enumerate(frequency_tokens['tokens']):
            # A conversion between namespaces (full vocab to stopless) is necessary.
            words = [self.vocab.get_token_from_index(index) for index in row.tolist()]
            word_counts = dict(Counter(words))
            num_words = sum(word_counts.values())

            # TODO: Make this faster.
            for word, count in word_counts.items():
                if word in self.tokens_to_index:
                    index = self.vocab.get_token_index(word, "stopless")

                    # Exclude padding token from influencing inference.
                    res[i][index] = (count * int(index > 0)) / num_words

        return res

    def _compute_stopword_mask(self, output_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """ Given a set of output tokens, compute a mask where 1 indicates stopword presence and 0
            indicates stopword absence.
        """
        res = torch.zeros_like(output_tokens['tokens'])
        for i, row in enumerate(output_tokens['tokens']):
            words = [self.vocab.get_token_from_index(index) for index in row.tolist()]
            res[i] = torch.LongTensor([int(word in STOP_WORDS) for word in words])

        return res

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # TODO: What makes sense for a decode for TopicRNN?
        return output_dict
