import logging
import pdb
import shutil
import time
from enum import Enum
from os import path
from typing import Dict, List

import pandas
import wget
from prepare_subset import open_local_article_set

"""
Module for going from raw article dicts to a dataframe
"""



# from Bio import Entrez
logging.basicConfig(level=logging.DEBUG)

ABSTRACT_PART = Enum(
    "ABSTRACT_PART", "background objective methods conclusions results invalid"
)


def load_structured_labels(datapath="data") -> Dict:
    """
    Return dictionnary mapping abtract labels to NLBM categories
    """
    logging.debug("Opening structured abstracts mappings file.")

    filename = "Structured-Abstracts-Labels-102615.txt"
    if not path.exists(path.join(datapath, filename)):
        logging.debug("No label file present. Downloading...")
        url = "https://structuredabstracts.nlm.nih.gov/Downloads/Structured-Abstracts-Labels-102615.txt"
        filename = wget.download(url)
        shutil.move(filename, path.join(datapath, filename))
        logging.debug("Finished downloading file.")

    file_path = path.join(datapath, filename)
    start = time.time()
    results = dict()
    results["origin_file"] = file_path
    with open(file_path, "r") as f:
        logging.debug("Opened file: %s", datapath)
        count = 0
        for line in f:
            parts = line.split(sep="|")
            assert parts, "Error: An empty line was read"
            results[parts[0].lower()] = ABSTRACT_PART[parts[1].lower()]
            count += 1
        logging.debug("Processed {} abstract labels.".format(count))
    logging.info(
        "Loaded structured labels from text (%fs).",
        round(time.time() - start, 2)
    )
    return results


def make_abstract_tuple(article_dict: Dict, category_mappings: Dict[str, str]) -> List:
    """
    Generate a list containing the article data

    article_dict: dictionary containing article information as returned by Bio.Entrez
    category_mappings: dictionary containing mappings from possible abstract labels to NLBM categories.
    """
    categories = {i: [] for i in ABSTRACT_PART}
    pmid = str(article_dict["MedlineCitation"]["PMID"])
    article = article_dict["MedlineCitation"]["Article"]
    date = dict(article_dict["MedlineCitation"]["DateCompleted"])

    # If the article is not in english don't add it to the dataframe
    lang = article["Language"][0]
    if lang != "eng":
        logging.debug("Article is not in english")
        return

    title = str(article["ArticleTitle"])
    if len(title) < 5:
        logging.error(
            "Error: Something went wrong (article title is too short)")
        return

    journal = str(article["Journal"]["Title"])
    if not journal:
        logging.debug("Error: Article journal's name is not defined")
        return

    # Detect correct section labels and add to corresponding category
    try:
        for section in article["Abstract"]["AbstractText"]:
            try:
                label = section.attributes["Label"].lower().strip()
            except AttributeError:
                logging.error(
                    "Section of article \"%s\" with pmid %s has no label.", title, pmid)
                continue
            try:
                correct_label = category_mappings[label]
            except KeyError:
                logging.error(
                    "Found unknown abstract label: '%s'.", label)
                correct_label = ABSTRACT_PART(6)
            if correct_label != ABSTRACT_PART(6):
                categories[correct_label].append(str(section))
    except KeyError:
        logging.error(
            "Article \"%s\" with PMID %s has no 'Abstract' key...", title, pmid)

        return

    result = ["\n".join(categories[cat]) for cat in categories.keys()]
    result.pop(-1)
    result.append(title)
    result.append(journal)
    result.append(pmid)
    result.append(date)
    return result


def convert_articles_to_dataframe(article_set_path: str, data_path: str) -> pandas.DataFrame:
    """
    Load a local article set and put it into a pandas dataframe.
    """
    start = time.time()
    arts = open_local_article_set(article_set_path)()

    cats = load_structured_labels(data_path)

    headers = ["background", "objective", "methods",
               "conclusions", "results", "title", "journal", "pmid", "date"]
    art_array = []

    for article in arts:
        art_array.append(make_abstract_tuple(article, cats))

    arts_nonull = [a for a in art_array if a is not None]
    df = pandas.DataFrame(arts_nonull, columns=headers)
    logging.debug("Loaded articles into data frame (%f s)", time.time()-start)
    # df.to_csv(output_path)
    return df
