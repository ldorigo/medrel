import logging
import pickle
import xml.etree.ElementTree as ET
import time

from Bio import Entrez

logging.basicConfig(level=logging.DEBUG)

Entrez.email = "ludor19@student.sdu.dk"


def get_pubmed_articles(query, name, id_batch_size=1000, article_batch_size=100):
    assert id_batch_size > article_batch_size, "Error: The article batch size has to be smaller than the ID batch size"
    assert id_batch_size <= 100000, "Error: Cannot get more than 100.000 ids at a time from pubmed."
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
                "{}_{}".format(name, currentarticles), "wb"))
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
                logging.info("ETA: {}m{}s".format(int(tot_time //60), int(tot_time%60)))

            articles_iteration_count += 1
            currentarticles += article_batch_size


query = """hasstructuredabstract[All Fields] AND medline[sb] AND "obesity"[All Fields] AND ("2009/03/20"[PDat] : "2019/03/17"[PDat] AND "humans"[MeSH Terms])"""


get_pubmed_articles(query, "test_run_1", 100000, 1000)
