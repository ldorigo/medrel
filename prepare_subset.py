"""
Module to handle downloading data from pubmed
"""
import logging
import gc
import pickle
import time
from os.path import exists, join, isfile
from os import makedirs, listdir
from Bio import Entrez
from shutil import rmtree
logging.basicConfig(level=logging.DEBUG)

Entrez.email = "ludor19@student.sdu.dk"


def get_pubmed_articles(query, name, id_batch_size=1000, article_batch_size=100):
    """
    Download a set of articles from pubmed.

    query: query to send to pubmed
    name: directory name in which to save the results. must not exist yet.
    id_batch_size: amount of id's to get at a time
    article_batch_size: amount of articles to download at a time.
    """
    assert id_batch_size >= article_batch_size, "Error: The article batch size has to be smaller than the ID batch size"
    assert id_batch_size <= 100000, "Error: Cannot get more than 100.000 ids at a time from pubmed."

    if exists(name):
        logging.error("Error: directory for supplied name already exists. ")
        choice = input("Overwrite? [n]")
        while choice not in ["y", "n", ""]:
            choice = input("Overwrite? [n]")
        if choice == "y":
            rmtree(name)
        else:
            return

    makedirs(name)
    logging.debug("Created directory for abstract files: %s", name)

    counthandle = Entrez.egquery(term=query)
    record = Entrez.read(counthandle)
    amount = 0
    for row in record["eGQueryResult"]:
        if row["DbName"] == "pubmed":
            amount = int(row["Count"])
            logging.info("Found {} articles.".format(amount))
    currentids = 0
    currentarticles = 0
    ids_iteration_count = 0
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
        if ids_iteration_count == 0:
            elapsed = time.time() - start
        logging.info("Fetched ids {} to {} (on a total of {}).".format(
            ids_iteration_count * id_batch_size,
            (ids_iteration_count + 1) * (id_batch_size),
            amount))
        currentids += id_batch_size
        ids_iteration_count += 1
        articles_iteration_count = 0
        while currentarticles < currentids:
            if ids_iteration_count == 1 and articles_iteration_count == 0:
                start = time.time()

            next_articles_ids = next_ids[articles_iteration_count * article_batch_size:(
                articles_iteration_count+1)*article_batch_size]
            next_articles_handle = Entrez.efetch(
                db="pubmed", id=next_articles_ids, retmode="xml")
            next_articles_record = Entrez.read(next_articles_handle)
            next_articles_handle.close()
            pickle.dump(next_articles_record, open(
                "{}/articles_{}".format(name, currentarticles), "wb"))
            logging.info("Fetched articles {} to {} (on a total of {}) from ids batch number {}".format(
                articles_iteration_count * article_batch_size,
                (articles_iteration_count + 1) * article_batch_size,
                len(next_ids),
                ids_iteration_count
            ))
            if ids_iteration_count == 1 and articles_iteration_count == 0:
                elapsed_rec = time.time() - start
                batch_time = elapsed_rec * id_batch_size / article_batch_size + elapsed
                tot_time = batch_time * (amount / id_batch_size)
                logging.info("ETA: {}m{}s".format(
                    int(tot_time // 60), int(tot_time % 60)))

            articles_iteration_count += 1
            currentarticles += article_batch_size


def open_local_article_set(directory: str):
    """
    Return a generator that allows to seamlessly iterate
    through local files containing pubmed articles.
    """
    assert exists(directory), "Error: supplied directory does not exist."

    logging.debug("Opening set of articles: %s.", directory)

    files = [
        join(directory, f)
        for f in listdir(directory)
        if isfile(join(directory, f)) and f != ".DS_Store"
    ]
    logging.debug("%d files in directory.", len(files))

    def load_file(file):
        with open(file, "rb") as f:
            gc.disable()
            res = pickle.load(f)
            gc.enable()
            return res

    def my_generator():
        for file in files:
            logging.debug("Opening next file: %s", file)
            with open(file, "rb") as f:
                current_list = load_file(file)
            for item in current_list["PubmedArticle"]:
                yield item

    return my_generator
