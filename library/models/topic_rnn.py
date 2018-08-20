from collections import Counter
from typing import Dict, Optional

import torch
import torch.nn as nn
from allennlp.common import Params
from allennlp.data.vocabulary import (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN,
                                      Vocabulary)
from allennlp.models.model import Model
from allennlp.modules import (FeedForward, Seq2SeqEncoder, TextFieldEmbedder,
                              TimeDistributed)
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from overrides import overrides
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import softmax
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
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 variational_autoencoder: FeedForward = None,
                 vae_hidden_size: int = 128,
                 topic_dim: int = 20,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TopicRNN, self).__init__(vocab, regularizer)

        # TODO: Sanity checks.

        # Perplexity for now is the only metric.
        # TODO: Categorical Accuracy for classification.
        self.metrics ={
            'perplexity': Perplexity()
        }

        self.text_field_embedder = text_field_embedder
        self.vocab_size = self.vocab.get_vocab_size("tokens")
        self.text_encoder = text_encoder
        self.vae_hidden_size = vae_hidden_size
        self.topic_dim = topic_dim
        self.vocabulary_projection_layer = TimeDistributed(Linear(text_encoder.get_output_dim(),
                                                                  self.vocab_size))

        # Compute stop indices in the normal vocab space to prevent stop words
        # from contributing to the topic additions.
        self.tokens_to_index = vocab.get_token_to_index_vocabulary()
        assert self.tokens_to_index[DEFAULT_PADDING_TOKEN] == 0 and self.tokens_to_index[DEFAULT_OOV_TOKEN] == 1
        for token, _ in self.tokens_to_index.items():
            if token not in STOP_WORDS:
                vocab.add_token_to_namespace(token, "stopless")

        self.stop_indices = torch.LongTensor([vocab.get_token_index(stop) for stop in STOP_WORDS])

        # Learnable topics.
        # TODO: How should these be initialized?
        self.beta = nn.Parameter(torch.rand(topic_dim, self.vocab_size))

        # mu: The mean of the variational distribution.
        self.w_mu = nn.Parameter(torch.rand(topic_dim))
        self.a_mu = nn.Parameter(torch.rand(topic_dim))

        # sigma: The root standard deviation of the variational distribution.
        self.w_sigma = nn.Parameter(torch.rand(topic_dim))
        self.a_sigma = nn.Parameter(torch.rand(topic_dim))

        # noise: used when sampling.
        self.noise = MultivariateNormal(torch.zeros(topic_dim), torch.eye(topic_dim))

        stopless_dim = vocab.get_vocab_size("stopless")
        self.variational_autoencoder = variational_autoencoder or FeedForward(
            # Takes as input the word frequencies in the stopless dimension and projects
            # the word frequencies into a latent topic representation.
            #
            # Each latent representation will help tune the variational dist.'s parameters.
            stopless_dim,
            2,
            [vae_hidden_size, topic_dim],
            torch.nn.ReLU(),
        )

        self.topic_dim = topic_dim

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                input_tokens: Dict[str, torch.LongTensor],
                output_tokens: Dict[str, torch.LongTensor],
                frequency_tokens: Optional[Dict[str, torch.LongTensor]] = None) -> Dict[str, torch.Tensor]:
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

        # Initial projection into vocabulary space.
        # Shape: (batch x sequence length x vocabulary size)
        logits = self.vocabulary_projection_layer(encoded_input)

        # Word frequency vectors and noise aren't generated with the model. If the model
        # is running on a GPU, these tensors need to be moved to the correct device.
        device = logits.device

        # Compute the topic additions when frequency tokens are available.
        if frequency_tokens:
            # 1. Compute noise for sampling.
            epsilon = self.noise.rsample().to(device=device)

            # 2. Compute Gaussian parameters.
            stopless_word_frequencies = self._compute_word_frequency_vector(frequency_tokens).to(device=device)
            mapped_term_frequencies = self.variational_autoencoder(stopless_word_frequencies)

            mu = self.w_mu * mapped_term_frequencies + self.a_mu
            log_sigma = self.w_sigma * mapped_term_frequencies + self.a_sigma

            # 3. Compute topic proportions given Gaussian parameters.
            theta = mu + torch.exp(log_sigma) * epsilon

            # Padding and OOV tokens are indexed at 0 and 1.
            topic_additions = torch.mm(theta, self.beta)
            topic_additions.t()[self.stop_indices] = 0  # Stop words have no contribution via topics.
            topic_additions.t()[0] = 0                  # Padding will be treated as stops.
            topic_additions.t()[1] = 0                  # Unknowns will be treated as stops.
            topic_additions = topic_additions.unsqueeze(1).expand_as(logits)
            logits += topic_additions

        if output_tokens:
            # Mask the output for proper loss calculation.
            output_mask = util.get_text_field_mask(output_tokens)
            relevant_output = output_tokens['tokens'].contiguous()
            relevant_output_mask = output_mask.contiguous()

            # Compute KL-Divergence.
            # A closed-form solution exists since we're assuming q is drawn
            # from a normal distribution.
            kl_divergence = 1 + 2 * log_sigma - (mu ** 2) - torch.exp(2 * log_sigma)

            # Sum along the topic dimension.
            kl_divergence = torch.sum(kl_divergence) / 2

            # Sampling log loss with L = 1
            # TODO: Should this be extended to support arbitrary L?
            cross_entropy_loss = util.sequence_cross_entropy_with_logits(logits, relevant_output,
                                                                         relevant_output_mask)

            loss = -kl_divergence + cross_entropy_loss
            output_dict['loss'] = loss

            # Compute perplexity.
            self.metrics['perplexity'](logits, relevant_output, relevant_output_mask)

        return output_dict

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

            # TODO: Make this faster.
            for word, count in word_counts.items():
                if word in self.tokens_to_index:
                    index = self.vocab.get_token_index(word, "stopless")

                    # Exclude padding token from influencing inference.
                    res[i][index] = count * int(index > 0)

        return res

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # TODO: What makes sense for a decode for TopicRNN?
        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'TopicRNN':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        text_encoder = Seq2SeqEncoder.from_params(params.pop("text_encoder"))
        topic_dim = params.pop("topic_dim")
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   text_encoder=text_encoder,
                   topic_dim=topic_dim,
                   initializer=initializer,
                   regularizer=regularizer)
