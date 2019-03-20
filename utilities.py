import logging
import pickle
import time
from enum import Enum
from os import listdir
from os.path import exists, isfile, join
logging.basicConfig(level=logging.DEBUG)

Abstract_Part = Enum(
    "Abstract_Part", "background objective methods conclusions results")


def open_local_article_set(directory):
    """
    Return a generator that allows to seamlessly iterate through local files containing pubmed articles.
    """
    assert exists(
        directory), "Error: supplied directory does not exist."

    logging.debug("Opening set of articles: {}.")

    files = [join(directory, f)
             for f in listdir(directory) if isfile(join(directory, f))]
    logging.debug("{} files in directory.".format(len(files)))

    def my_generator():
        for file in files:
            with open(file, "rb") as f:
                current_list = pickle.load(f)
            for item in current_list['PubmedArticle']:
                yield item

    return my_generator


def load_structured_labels(path):
    logging.debug("Opening structured abstracts mappings file.")
    start = time.time()
    results = dict()
    with open(path, 'r') as f:
        logging.debug("Opened file: {}".format(path))
        count = 0
        for line in f:
            parts = line.split(sep="|")
            assert parts, "Error: An empty line was read"
            results[parts[0].lower()] = Abstract_Part[parts[1].lower()]
            count += 1
        logging.debug("Processed {} abstract labels.".format(count))
    logging.info(
        "Loaded structured labels from text ({}s).".format(round(time.time()-start, 2)))
    return results


load_structured_labels("./Structured-Abstracts-Labels-102615.txt")
