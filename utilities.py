import logging
import pickle
import time
from enum import Enum
from os import listdir
from os.path import exists, isfile, join
import gc

logging.basicConfig(level=logging.DEBUG)

Abstract_Part = Enum(
    "Abstract_Part", "background objective methods conclusions results invalid"
)


def open_local_article_set(directory):
    """
    Return a generator that allows to seamlessly iterate 
    through local files containing pubmed articles.
    """
    assert exists(directory), "Error: supplied directory does not exist."

    logging.debug("Opening set of articles: {}.".format(directory))

    files = [
        join(directory, f)
        for f in listdir(directory)
        if isfile(join(directory, f)) and f != ".DS_Store"
    ]
    logging.debug("{} files in directory.".format(len(files)))

    def load_file(file):
        with open(file, "rb") as f:
            gc.disable()
            return pickle.load(f)
            gc.enable()
            
    def my_generator():
        for file in files:
            logging.debug("Opening next file: {}".format(file))
            with open(file, "rb") as f:
                current_list = load_file(file)
            for item in current_list["PubmedArticle"]:
                yield item

    return my_generator


def load_structured_labels(path):
    logging.debug("Opening structured abstracts mappings file.")
    start = time.time()
    results = dict()
    results["origin_file"] = path
    with open(path, "r") as f:
        logging.debug("Opened file: {}".format(path))
        count = 0
        for line in f:
            parts = line.split(sep="|")
            assert parts, "Error: An empty line was read"
            results[parts[0].lower()] = Abstract_Part[parts[1].lower()]
            count += 1
        logging.debug("Processed {} abstract labels.".format(count))
    logging.info(
        "Loaded structured labels from text ({}s).".format(
            round(time.time() - start, 2)
        )
    )
    return results


def add_structured_label(text, category, mappings):
    path = mappings["origin_file"]
    with open(path, "a") as f:
        f.write("\n{}|{}|".format(text, category.name))
    mappings[text] = category
    logging.debug(
        "Added category mapping {} to {} (in {})".format(text, category, path)
    )

