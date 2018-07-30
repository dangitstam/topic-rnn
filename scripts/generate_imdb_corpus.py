import argparse
import json
import os
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

    print("Producing unlabeled training jsonl:")
    train_unsup_out = os.path.join(args.save_dir, "train_unlabeled.jsonl")
    directory_to_jsonl(train_unsup_dir, train_unsup_out)

    print("Producing labeled training jsonl:")
    train_out = os.path.join(args.save_dir, "train.jsonl")
    directory_to_jsonl(train_pos_dir, train_out)
    directory_to_jsonl(train_neg_dir, train_out)

    print("Producing labeled testing jsonl:")
    test_out = os.path.join(args.save_dir, "test.jsonl")
    directory_to_jsonl(test_pos_dir, test_out)
    directory_to_jsonl(test_neg_dir, test_out)


def directory_to_jsonl(data_dir, save_path):
    """ 
    Given a directory containing training instances from the IMDB
    dataset, produces a jsonl with an example on each line of the form
    { "id": int, "text": str, "sentiment": int (optional) }

    :param data_dir: The directory containing the data instances.
    :param save_path: The path in which to save the jsonl.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    out_file = open(save_path, "a")
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
            json.dump(example, out_file, ensure_ascii=False)
            out_file.write('\n')


if __name__ == "__main__":
    main()
