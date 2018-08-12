import argparse
import json
import os
import random
import sys

from tqdm import tqdm


def main():
    """
    A tool for pre-processing the 100k IMDB movie review dataset.

    Given the path to the dataset (a path to the directory as a result of
    undoing the tar from from http://ai.stanford.edu/~amaas/data/sentiment/),
    produces three jsonl files: train.jsonl (training instances with sentiment,
    test.jsonl (testing instances with sentiment, and train_unlabeled.jsonl
    (training instances without sentiment).

    Expected structure
    data_path/
        train/
            pos/
            neg/
            unsup/
        test/
            pos/
            neg/
    
    Each line will be an example of the form:
    {
      "id": The unique ID given to each review,
      "text": The raw text of the review.
      "sentiment": The label for the review, either 1 (positive), 0 (negative),
                   or None (unlabeled).
    }

    Training and testing instances that are labeled will not come shuffled, and
    will appear positive and then negative. It is up to the dataset reader to
    shuffle them.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(  # Escape out into project directory.
            os.path.dirname( # Escape out into scripts directory.
                os.path.realpath(__file__))))))
    parser.add_argument("--data-path", type=str,
                        help="Path to the IMDB dataset directory.")
    parser.add_argument("--save-dir", type=str,
                        default=project_root,
                        help="Directory to store the preprocessed corpus.")
    parser.add_argument("--seed", type=int,
                        default=1337,
                        help="Random seed to use when shuffling data.")
    args = parser.parse_args()

    try:
        if os.path.exists(args.save_dir):
            input("IMDB corpus {} already exists.\n"
                  "Press <Ctrl-c> to exit or "
                  "<Enter> to recreate it.".format(args.save_dir))
    except KeyboardInterrupt:
        print()
        sys.exit()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # Path to unlabeled training directory.
    train_unsup_dir = os.path.join(args.data_path, "train", "unsup")
    assert os.path.exists(train_unsup_dir)

    # Paths to the labeled training directories.
    train_pos_dir = os.path.join(args.data_path, "train", "pos")
    train_neg_dir = os.path.join(args.data_path, "train", "neg")
    assert os.path.exists(train_pos_dir)
    assert os.path.exists(train_neg_dir)

    # Paths to the labeled testing directories.
    test_pos_dir = os.path.join(args.data_path, "test", "pos")
    test_neg_dir = os.path.join(args.data_path, "test", "neg")
    assert os.path.exists(test_pos_dir)
    assert os.path.exists(test_neg_dir)

    print("Parsing unlabeled training data:")
    train_unsup_examples = directory_to_jsons(train_unsup_dir)
    assert len(train_unsup_examples) == 50000

    print("Parsing labeled training data:")
    train_examples = directory_to_jsons(train_pos_dir)
    train_examples += directory_to_jsons(train_neg_dir)
    assert len(train_examples) == 25000

    print("Parsing labeled testing data:")
    test_examples = directory_to_jsons(test_pos_dir)
    test_examples += directory_to_jsons(test_neg_dir)
    assert len(test_examples) == 25000

    # In the paper, they use a combined set 65k training examples (labeled and
    # unlabeled) for the unsupervised portion of training.
    #
    # They then train a separate classifier for interpreting the results of the
    # final hidden state output of the TopicRNN into positive or negative sentiment.
    #
    # "train_unsup.jsonl" will include all 50k unlabeled samples with 10K randomly selected
    # labeled examples. The remaining labeled samples will be used for validation.
    #
    # A full version of the labeled training data will also be saved for training the classifier
    # (20K training, 5K validation).
    train_unsup_out = os.path.join(args.save_dir, "train_unsup.jsonl")
    valid_unsup_out = os.path.join(args.save_dir, "valid_unsup.jsonl")
    train_out = os.path.join(args.save_dir, "train_labeled.jsonl")
    valid_out = os.path.join(args.save_dir, "valid_labeled.jsonl")
    test_out = os.path.join(args.save_dir, "test.jsonl")

    # Shuffle training and take the first 15K examples.
    random.Random(args.seed).shuffle(train_examples)
    train_unsup_examples += train_examples[:15000]
    assert len(train_unsup_examples) == 65000

    # Remainder to serve as validation.
    valid_unsup_examples = train_examples[15000:]
    assert len(valid_unsup_examples) == 10000

    print("Saving training and validation unsupervised examples:")
    write_jsons_to_file(train_unsup_examples, train_unsup_out)
    write_jsons_to_file(valid_unsup_examples, valid_unsup_out)

    print("Saving training and valdiation labeled examples:")
    write_jsons_to_file(train_examples[:20000], train_out)
    write_jsons_to_file(train_examples[20000:], valid_out)

    print("Saving test labeled examples:")
    write_jsons_to_file(test_examples, test_out)


def directory_to_jsons(data_dir):
    """
    Given a directory containing training instances from the IMDB
    dataset, produces a list of json objects of the form
    { "id": int, "text": str, "sentiment": int (optional) }

    :param data_dir: The directory containing the data instances.
    :param save_path: The path in which to save the jsonl.
    """

    jsons = []
    for path in tqdm(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, path)

        # File names are expected to be XXXX_XX.txt
        [example_id, example_sentiment] = path.split('.')[0].split('_')

        # Write the example on it's own line.
        with open(full_path, 'r') as file:
            example_text = file.read()
            example = {
                "id": int(example_id),
                "text": example_text,
                "sentiment": int(example_sentiment)
            }
            jsons.append(example)

    return jsons


def write_jsons_to_file(jsons, save_path):
    """
    Write each json object in 'jsons' as its own line in the file designated by 'save_path'.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    out_file = open(save_path, "a")
    for example in tqdm(jsons):
        json.dump(example, out_file, ensure_ascii=False)
        out_file.write('\n')


if __name__ == "__main__":
    main()
