# TopicRNN: A Recurrent Neural Network with Long-Range Semantic Dependency
Implementation of Dieng et al.'s [TopicRNN](https://arxiv.org/abs/1611.01702): a neural topic model & RNN hybrid that learns global semantic dependencies via latent topics and local, syntatic dependencies through an RNN.

## Model Details
* The model learns a `beta` matrix of size `(V x K)` where `V` is the size of the vocabulary and `K` is the number of latent topics. Each row in `beta` represents a distinct distribution over the vocabulary.
* A variational distribution is learned using word frequencies as input to produce the parameters for the Gaussian distribution in which each topic proportion vector `theta` of length `k` is sampled from.
* `beta * theta` then results in the the logits over the vocabulary at the given time step that allow learned topics to be properly weighted before influencing inference of the next word. Topic additions for each word are zeroed out if the index of the logit belongs to a stop word, this allows only semantically significant words to have influence from the topics.
* The topic additions`beta * theta` are added to the vocabulary projection of the RNN hidden `W * ht` resulting in a final distribution over the vocabulary that is normalized via SoftMax.
* Loss is computed via the sum of the cross entropy loss between the predicted words and the actual words, along with the closed form KL-Divergence of the variational distribution's Gaussian parameters.

## Getting Started

### Generating a Dataset (IMDB)
`imdb_review_reader.py` contains a dataset reader primed to take a `.jsonl` file where each entry is of the form
```
{
    'id': <integer id>,
    'text': <raw text of movie review>,
    'sentiment': <integer value representing sentiment>
}
```

You can download the IMDB 100K dataset [here]( http://ai.stanford.edu/~amaas/data/sentiment/).

Upon extracting the dataset from the tar, the resulting directory will look like
```
aclImdb/
    train/
        unsup/
            <review id>_<sentiment>.txt
            ...
        pos/
            <review id>_<sentiment>.txt
            ...
        neg/
            <review id>_<sentiment>.txt
            ...
    test/
        pos/
            <review id>_<sentiment>.txt
            ...
        neg/
            <review id>_<sentiment>.txt
            ...
    ...
```

You can generate the necessary `.jsonl` files via `scripts/generate_imdb_corpus.py` which expects the `aclImdb` file structure above:

```
python generate_imdb_corpus.py --data-path <path to aclImdb>  --save-dir <directory to save the .jsonl files>
```

The directory specified by `--save-dir` will then contain three files: `train.jsonl`, `train_unlabeled.jsonl`, and `test.jsonl`. You will need to write the relative path to training/testing `.jsonl` files within your experiment JSON config.

### Training the model

`imdb_language_model.json`, a "cheating" version of a language model (cheating as in, the variational autoencoder is allowed to see the entire review before making inference), contains a base specification for TopicRNN (i.e hyperparamters, relative paths to training/testing `.jsonl`, etc.)

In this file, you must specify at minimum
* The dataset reader with `type` (i.e. `imdb_review_reader`) and `words_per_instance` (backpropagation-through-time limit)
* The relative paths to the training and validation `.jsonl` files (`generate_imdb_corpus.py` will be extended to produce training and validation splits at a later time)
* Vocabulary with `max_vocab_size`
* The model with `type` (base implementation of `topic_rnn` is currently the only model), `text_field_embedder` (specify whether to use pretrained embeddings, embedding size, etc.), `text_encoder` (encoding the utterance via RNN, GRU, LSTM, etc.), and `topic_dim` (number of latent topics)

To train the model, run
```
allennlp train <path to the current experiment's JSON configuration> \
-s <directory for serialization>  \
--include-package library
```

## Built With

* [AllenNLP](https://allennlp.org/) - The NLP framework used, built by AI2
* [PyTorch](https://pytorch.org/) - The deep learning library used

## Authors

* **Tam Dang**

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE) file for details.
