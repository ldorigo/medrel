"""
Module to handle downloading data from pubmed
"""
import gc
import logging
import pickle
import random
import shutil
import time
from dateutil import parser as dateparser
from enum import Enum
from os import listdir, makedirs
from os.path import exists, isfile, join
from shutil import rmtree
from typing import Dict, Generator, List, Optional

import ipywidgets as widgets
import numpy as np
import pandas
import wget
from Bio import Entrez
from tqdm.auto import tqdm

from utils import DATALOCATION_WIDGET, SESSIONLOCATION_WIDGET, get_logger

Entrez.email = "ludor19@student.sdu.dk"

logger = get_logger("prepare_subset")

class ABSTRACT_PART(Enum):
    background = 0
    objective = 1
    methods = 2
    conclusions = 3
    results = 4
    invalid = 5



# ABSTRACT_ENUM = Enum("ABSTRACT_ENUM", "background objective methods conclusions results invalid")

def get_downloaded_articles_path() -> str:
    return join(DATALOCATION_WIDGET.value, "sessions", SESSIONLOCATION_WIDGET.value, "downloaded_articles")


DEFAULT_QUERY = """hasstructuredabstract[All Fields] AND medline[sb] AND "2009/03/20"[PDat] : "2019/03/17"[PDat] AND "humans"[MeSH Terms]"""
QUERY_WIDGET = widgets.Textarea(value=DEFAULT_QUERY, disabled=False, rows=12)


def get_pubmed_articles(query: str, name: str, id_batch_size=1000, article_batch_size=100, max_amount=-1, sample=False) -> None:
    """Get a set of articles from pubmed that correspond to the given query.

    Args:
        query (str): query to send to pubmed
        name (str): folder name to which to save the set
        id_batch_size (int, optional): Amount of ids to get at a time. Maximum 100.000. Defaults to 1000.
        article_batch_size (int, optional): Amount of articles to get at a time. Defaults to 100.
        max_amount (int, optional): Maximum amount of articles to download. Defaults to -1 for no limit.
        sample (bool, optional): if amount is set, get a random sample instead of the first n results. Defaults to False.
    """
    assert id_batch_size <= 100000, "Error: Cannot get more than 100.000 ids at a time from pubmed."

    if exists(name):
        logger.error(
            "Error: directory for supplied name ({}) already exists.".format(name))
        choice = input("Overwrite? [n]")
        while choice not in ["y", "n", ""]:
            choice = input("Overwrite? [n]")
        if choice == "y":
            rmtree(name)
        else:
            return

    makedirs(name)
    logger.debug(
        "Created directory for abstract files: %s, sending query.", name)
    counthandle = Entrez.egquery(term=query)
    record = Entrez.read(counthandle)
    amount = 0
    for row in record["eGQueryResult"]:
        if row["DbName"] == "pubmed":
            amount = int(row["Count"])
            logger.info("Found {} articles.".format(amount))
    currentids = 0
    currentarticles = 0
    ids_iteration_count = 0
    ids: List[str]
    ids = []

    while currentids < amount:
        if ids_iteration_count == 0:
            start = time.time()
        # We get the next batch of ids:
        next_ids_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=id_batch_size,
            retstart=id_batch_size * ids_iteration_count
        )

        next_ids_record = Entrez.read(next_ids_handle)
        next_ids_handle.close()
        next_ids = next_ids_record["IdList"]
        ids.extend(next_ids)
        if ids_iteration_count == 0:
            elapsed = time.time() - start
        logger.info("Fetched ids {} to {} (on a total of {}).".format(
            ids_iteration_count * id_batch_size,
            (ids_iteration_count + 1) * (id_batch_size),
            amount))
        currentids += id_batch_size
        ids_iteration_count += 1

    if 0 < max_amount < amount:
        if random:
            ids = random.sample(ids, max_amount)
            logger.info("Selected {} random articles.".format(max_amount))
        else:
            ids = ids[0:max_amount]
            logger.info("Selected {} first articles.".format(max_amount))

    articles_iteration_count = 0
    logger.info("Sending request for first batch of articles...")
    while currentarticles < len(ids):
        if articles_iteration_count == 0:
            start = time.time()

        next_articles_ids = ids[articles_iteration_count * article_batch_size:(
            articles_iteration_count + 1) * article_batch_size]

        next_articles_handle = Entrez.efetch(
            db="pubmed", id=next_articles_ids, retmode="xml")
        next_articles_record = Entrez.read(next_articles_handle,validate=False)
        next_articles_handle.close()

        pickle.dump(next_articles_record, open(
            "{}/articles_{}".format(name, currentarticles), "wb"))

        logger.info("Fetched articles {} to {} (on a total of {})".format(
            articles_iteration_count * article_batch_size,
            (articles_iteration_count + 1) * article_batch_size,
            len(ids)))
        if articles_iteration_count == 0:
            elapsed_rec = time.time() - start
            tot_time = elapsed_rec * (len(ids) / article_batch_size)
            logger.info("ETA: {}m{}s".format(
                int(tot_time // 60), int(tot_time % 60)))

        articles_iteration_count += 1
        currentarticles += article_batch_size
    logger.info("Done!")




def open_local_article_set(directory: str) -> Generator: 
    """
    Return a generator that allows to seamlessly iterate
    through local files containing pubmed articles.
    """

    assert exists(directory), "Error: supplied directory does not exist."

    logger.debug("Opening set of articles: %s.", directory)

    files = [
        join(directory, f)
        for f in listdir(directory)
        if isfile(join(directory, f)) and f != ".DS_Store"
    ]
    logger.debug("%d files in directory.", len(files))

    def load_file(file):
        with open(file, "rb") as f:
            gc.disable()
            res = pickle.load(f)
            gc.enable()
            return res

    for file in files:
        logger.debug("Opening next file: %s", file)
        with open(file, "rb") as f:
            current_list = load_file(file)
        for item in current_list["PubmedArticle"]:
            yield item



def load_structured_labels(datapath="data") -> Dict[str,str]:
    """
    Return dictionnary mapping abtract labels to NLBM categories
    """
    logger.debug("Opening structured abstracts mappings file.")

    filename = "Structured-Abstracts-Labels-102615.txt"
    if not exists(join(datapath, filename)):
        logger.info("No label file present. Downloading...")
        url = "https://structuredabstracts.nlm.nih.gov/Downloads/Structured-Abstracts-Labels-102615.txt"
        filename = wget.download(url)
        shutil.move(filename, join(datapath, filename))
        logger.info("Finished downloading file.")

    file_path = join(datapath, filename)
    start = time.time()
    results: Dict[str, str]
    results = dict()
    results["origin_file"] = file_path
    with open(file_path, "rt") as f:
        logger.debug("Opened file: %s", datapath)
        count = 0
        for line in f:
            parts: List[str]
            parts = line.split(sep="|")
            assert parts, "Error: An empty line was read"
            specific_name = parts[0].lower()
            generic_name = parts[1].lower()

            results[specific_name] = generic_name
            assert generic_name in [a.name for a in list(ABSTRACT_PART)], "Error: not an abstract part"

            count += 1
        logger.debug("Processed {} abstract labels.".format(count))
    logger.info(
        "Loaded structured labels from text (%fs).",
        round(time.time() - start, 2)
    )
    return results


def make_abstract_tuple(article_dict: Dict, category_mappings: Dict[str, str]) -> Optional[List]:
    """
    Generate a list containing the article data

    article_dict: dictionary containing article information as returned by Bio.Entrez
    category_mappings: dictionary containing mappings from possible abstract labels to NLBM categories.
    """
    categories: Dict[ABSTRACT_PART, List[str]]
    categories = {i: [] for i in list(ABSTRACT_PART)}
    pmid = str(article_dict["MedlineCitation"]["PMID"])
    article = article_dict["MedlineCitation"]["Article"]
    try:
        date = dict(article_dict["MedlineCitation"]["DateCompleted"])
    except KeyError:
        logging.debug("Error: No date was found")
        date = {}
    # If the article is not in english don't add it to the dataframe
    lang = article["Language"][0]
    if lang != "eng":
        logger.debug("Article is not in english")
        return None

    title = str(article["ArticleTitle"])
    if len(title) < 5:
        logger.debug("Error: Something went wrong (article title is too short)")
        return None

    journal = str(article["Journal"]["Title"])
    if not journal:
        logger.debug("Error: Article journal's name is not defined")
        return None

    # Detect correct section labels and add to corresponding category
    try:
        for section in article["Abstract"]["AbstractText"]:
            try:
                label : str
                label = section.attributes["Label"].lower().strip()
            except AttributeError:
                logger.debug(
                    "Section of article \"%s\" with pmid %s has no label.", title, pmid)
                continue
            try:
                cat = category_mappings[label]
                correct_label = ABSTRACT_PART[cat]
            except KeyError:
                logger.debug(
                    "Found unknown abstract label: '%s'.", label)
                correct_label = ABSTRACT_PART.invalid
            if correct_label is not ABSTRACT_PART.invalid:
                categories[correct_label].append(str(section))
    except KeyError:
        logger.debug(
            "Article \"%s\" with PMID %s has no 'Abstract' key...", title, pmid)
        return None

    result = ["\n".join(categories[cat]) for cat in categories.keys()]
    result.pop(-1)
    result.append(title)
    result.append(journal)
    result.append(pmid)
    result.append(parse_dates(date))
    return result


def convert_articles_to_dataframe(article_set_path: str, data_path: str) -> pandas.DataFrame:
    """
    Load a local article set and put it into a pandas dataframe.
    """
    start = time.time()

    arts = open_local_article_set(article_set_path)

    cats = load_structured_labels(data_path)

    headers = ["background", "objective", "methods",
               "conclusions", "results", "title", "journal", "pmid", "date"]
    art_array = []

    for article in tqdm(arts):
        art_array.append(make_abstract_tuple(article, cats))

    arts_nonull = [a for a in art_array if a is not None]
    df = pandas.DataFrame(arts_nonull, columns=headers)
    logger.info("Loaded articles into data frame (%f s)", time.time()-start)
    return df.replace(np.nan, '', regex=True)


def parse_dates(date_dict: Dict[str, str]) -> str:
    try:
        datestr = date_dict["Month"] + "." + \
            date_dict["Day"] + "." + date_dict["Year"]
        date = dateparser.parse(datestr).timestamp()
    except KeyError:
        date = 0
    return str(date)


if __name__ == "__main__":
    import doctest
    doctest.testmod()


