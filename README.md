# TopicRNN: A Recurrent Neural Network with Long-Range Semantic Dependency
Implementation of Dieng et al.'s [TopicRNN](https://arxiv.org/abs/1611.01702): a neural topic model & RNN hybrid that learns global semantic dependencies via latent topics and local, syntatic dependencies through an RNN.

## Model Details
* The model learns a `beta` matrix of size `(V x K)` where `V` is the size of the vocabulary and `K` is the number of latent topics. Each row in `beta` represents a distinct distribution over the vocabulary.
* A variational distribution is learned using word frequencies as input to produce the parameters for the Gaussian distribution in which each topic proportion vector `theta` of length `k` is sampled from.
* `beta * theta` then results in the the logits over the vocabulary at the given time step that allow learned topics to be properly weighted before influencing inference of the next word. Topic additions for each word are zeroed out if the index of the logit belongs to a stop word, this allows only semantically significant words to have influence from the topics.
* The topic additions`beta * theta` are added to the vocabulary projection of the RNN hidden `W * ht` resulting in a final distribution over the vocabulary that is normalized via SoftMax.
* Loss is computed via the sum of the cross entropy loss between the predicted words and the actual words, along with the closed form KL-Divergence of the variational distribution's Gaussian parameters.

## Getting Started

### Generating a Dataset


### Training the model
