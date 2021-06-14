"""
Module to handle downloading data from pubmed
"""
import dataclasses
import gc
import json
import logging
import os
import pickle
import random
import shutil
import time
from enum import Enum
from os import listdir, makedirs
from os.path import exists, isfile, join
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import spacy
import wget
from Bio import Entrez

from lib.defaults import default_data
from lib.utils import date_to_timestamp, get_logger

Entrez.email = "ludor19@student.sdu.dk"

_logger = get_logger("prepare_subset")


class ABSTRACT_PART(Enum):
    background = 0
    objective = 1
    methods = 2
    conclusions = 3
    results = 4
    invalid = 5


def set_extensions():
    """Add relevant extensions to spacy docs and spans"""
    if not spacy.tokens.Doc.has_extension("journal"):
        spacy.tokens.Doc.set_extension("journal", default="")
    if not spacy.tokens.Doc.has_extension("title"):
        spacy.tokens.Doc.set_extension("title", default="")
    if not spacy.tokens.Doc.has_extension("pmid"):
        spacy.tokens.Doc.set_extension("pmid", default=0)
    if not spacy.tokens.Doc.has_extension("date"):
        spacy.tokens.Doc.set_extension("date", default=0)
    if not spacy.tokens.Span.has_extension("CUI"):
        spacy.tokens.Span.set_extension("CUI", default="UNKNOWN")
    if not spacy.tokens.Span.has_extension("abstract_part"):
        spacy.tokens.Span.set_extension("abstract_part", default="NONE")


@dataclasses.dataclass
class DataSetMetadata:
    query: str
    matched_pmids: List[str]
    selected_pmids: List[str]
    pmids_processed: List[str]
    sample: bool = False
    max_amount: int = -1


class DataSet:
    def __init__(
        self, path: str, query: str = "", max_size: int = -1, sample: bool = False
    ):
        """Initialize a DataSet or load from disk.

        If the dataset was created before, load it from the given path. Otherwise create it and associate it with the specified query.

        Args:
            path (str): Path of the dataset
            query ([type], optional): If new dataset, the pubmed query that is associated to it. Defaults to None.
            max_size (int, optional): Maximum number of abstracts contained in the dataset. Defaults to -1 for no limit.
            sample (bool, optional): whether to draw a random sample.
        Raises:
            AssertionError: If a query is provided for an existing dataset, or inversely.
        """

        self.path: Path = Path(path)
        self.metadata: DataSetMetadata
        self.nlp = None
        self.abstract_labels = None
        if "metadata.json" not in os.listdir(path):
            _logger.debug("Dataset not initialized. Creating metadata file.")
            assert query != "", "Uninitialized dataset with no query provided."
            self.metadata = DataSetMetadata(
                query=query,
                matched_pmids=[],
                selected_pmids=[],
                pmids_processed=[],
                sample=sample,
                max_amount=max_size,
            )
            self.serialize()

        else:
            self.metadata = self.deserialize()

    def serialize(self):
        with open(self.path / "metadata.json", "w") as md_file:
            json.dump(self.metadata.__dict__, md_file)

    def deserialize(self) -> DataSetMetadata:
        with open(self.path / "metadata.json", "r") as f:
            raw_dict = json.load(f)

        return DataSetMetadata(**raw_dict)

    def fetch_ids(self, batch_size: int = 10000) -> None:
        """Get the list of pubmed ids corresponding to the query. Save them as an attribute and return them.

        Args:
            batch_size (int, optional): Number of ids to get at a time.. Defaults to 10000.
        Returns:
            List[str]: list of pubmed ids corresponding to the query.
        """
        _logger.debug("Fetching ids corresponding to query: ")
        _logger.debug(self.metadata.query)

        ids_handle = Entrez.egquery(term=self.metadata.query)
        record: Dict
        record = Entrez.read(ids_handle)
        amount = 0
        for row in record["eGQueryResult"]:
            if row["DbName"] == "pubmed":
                amount = int(row["Count"])
                _logger.info("Found {} ids.".format(amount))

        downloaded_ids_count = 0
        #         # currentabstracts = 0
        ids_iteration_count = 0
        ids: List[str] = []

        while downloaded_ids_count < amount:
            #             # if ids_iteration_count == 0:
            start = time.time()
            _logger.debug("Fetching id list...")
            # We get the next batch of ids:
            next_ids_handle = Entrez.esearch(
                db="pubmed",
                term=self.metadata.query,
                retmax=batch_size,
                retstart=batch_size * ids_iteration_count,
            )

            next_ids_record = Entrez.read(next_ids_handle)
            next_ids_handle.close()
            next_ids: List[str] = next_ids_record["IdList"]
            ids.extend(next_ids)
            elapsed = time.time() - start

            _logger.info(
                "Fetched ids {} to {} (on a total of {:.2f}). ({}s)".format(
                    ids_iteration_count * batch_size,
                    (ids_iteration_count + 1) * (batch_size),
                    amount,
                    elapsed,
                )
            )

            downloaded_ids_count += batch_size
            ids_iteration_count += 1

        _logger.info("Fetched {} ids. ".format(len(ids)))
        # self.metadata[""]
        self.metadata.matched_pmids = ids

        # Sample or reduce size
        if self.metadata.max_amount != -1 and self.metadata.max_amount < len(ids):
            if self.metadata.sample:
                self.metadata.selected_pmids = random.sample(
                    ids, self.metadata.max_amount
                )
            else:
                self.metadata.selected_pmids = ids[: self.metadata.max_amount]
        else:
            self.metadata.selected_pmids = ids

    def get_raw_abstracts_generator(
        self,
        id_batch_size: int = 10000,
        abstract_batch_size: int = 100,
        restart: bool = False,
    ) -> Generator[Any, None, None]:
        """Get a generator that yields the abstracts corresponding to the dataset's query from pubmed.
        Args:
            id_batch_size (int, optional): Amount of ids to get at a time. Maximum 100.000. Defaults to 1000.
            abstract_batch_size (int, optional): Amount of abstracts to get at a time. Defaults to 100.
            force (bool, optional): force download even if already downloaded. Defaults to False.
        """
        assert (
            id_batch_size <= 100000
        ), "Error: Cannot get more than 100.000 ids at a time from pubmed."
        assert (
            self.metadata.matched_pmids
        ), "Error: trying to get abstracts before getting list of pmids"

        abstracts_iteration_count = 0
        _logger.info("Sending request for first batch of abstracts...")
        downloaded_abstract_count: int = 0
        ids = self.metadata.selected_pmids
        while downloaded_abstract_count < len(ids):
            start = time.time()

            next_abstracts_ids = ids[
                abstracts_iteration_count
                * abstract_batch_size : (abstracts_iteration_count + 1)
                * abstract_batch_size
            ]

            next_abstracts_handle = Entrez.efetch(
                db="pubmed", id=next_abstracts_ids, retmode="xml"
            )
            next_abstracts_record = Entrez.read(next_abstracts_handle, validate=False)
            next_abstracts_handle.close()

            _logger.info(
                "Fetched abstracts {} to {} (on a total of {})".format(
                    abstracts_iteration_count * abstract_batch_size,
                    (abstracts_iteration_count + 1) * abstract_batch_size,
                    len(ids),
                )
            )
            # elapsed_rec = time.time() - start
            # tot_time: float = elapsed_rec * (
            #     (len(ids) / abstract_batch_size) - abstracts_iteration_count
            # )

            for item in next_abstracts_record["PubmedArticle"]:
                yield item

            abstracts_iteration_count += 1
            downloaded_abstract_count += abstract_batch_size

        _logger.info("Done!")

    def load_structured_labels(self, datapath: str = "data") -> Dict[str, str]:
        """
        Return dictionnary mapping abtract labels to NLBM categories
        """
        _logger.debug("Opening structured abstracts mappings file.")

        filename = "Structured-Abstracts-Labels-102615.txt"
        if not (Path(datapath) / filename).exists():
            _logger.info("No label file present. Downloading...")
            url = "https://structuredabstracts.nlm.nih.gov/Downloads/Structured-Abstracts-Labels-102615.txt"
            filename = wget.download(url)
            shutil.move(filename, join(datapath, filename))
            _logger.info("Finished downloading file.")

        file_path = join(datapath, filename)
        start = time.time()
        results: Dict[str, str]
        results = dict()
        results["origin_file"] = file_path
        with open(file_path, "rt") as f:
            _logger.debug("Opened file: %s", datapath)
            count = 0
            for line in f:
                parts: List[str]
                parts = line.split(sep="|")
                assert parts, "Error: An empty line was read"
                specific_name = parts[0].lower()
                generic_name = parts[1].lower()

                results[specific_name] = generic_name
                assert generic_name in [
                    a.name for a in list(ABSTRACT_PART)
                ], "Error: not an abstract part"

                count += 1
            _logger.debug("Processed {} abstract labels.".format(count))
        _logger.info(
            "Loaded structured labels from text (%fs).", round(time.time() - start, 2)
        )
        self.abstract_labels = results
        return results

    def init_spacy(self):
        start = time.time()
        self.nlp = spacy.load("en_core_sci_scibert")
        set_extensions()
        _logger.debug("Loaded spacy model ({}s)".format(time.time() - start))

    def make_abstract_doc(self, abstract_dict: Dict) -> Optional[List]:
        """
        Generate a spacy doc containing the abstract data.

        abstract_dict: dictionary containing abstract information as returned by Bio.Entrez
        category_mappings: dictionary containing mappings from possible abstract labels to NLBM categories.
        """
        # if self.abstract_labels  None:
        #     self.load_structured_labels()
        if self.nlp is None:
            self.init_spacy()

        categories: Dict[ABSTRACT_PART, List[str]]
        categories = {i: [] for i in list(ABSTRACT_PART)}
        pmid = str(abstract_dict["MedlineCitation"]["PMID"])
        abstract = abstract_dict["MedlineCitation"]["Article"]
        try:
            date = date_to_timestamp(
                dict(abstract_dict["MedlineCitation"]["DateCompleted"])
            )
        except KeyError:
            logging.debug("Error: No date was found")
            date = 0
        # If the abstract is not in english don't consider it
        lang = abstract["Language"][0]
        if lang != "eng":
            _logger.debug("Article is not in english")
            return None

        title = str(abstract["ArticleTitle"])
        if len(title) < 5:
            _logger.debug("Error: Something went wrong (abstract title is too short)")
            return None

        journal = str(abstract["Journal"]["Title"])
        if not journal:
            _logger.debug("Error: Article journal's name is not defined")
            return None
        results = [str(section) for section in abstract["Abstract"]["AbstractText"]]
        # Detect correct section labels and add to corresponding category
        # try:
        #     for section in abstract["Abstract"]["AbstractText"]:
        #         try:
        #             label: str
        #             label = section.attributes["Label"].lower().strip()
        #         except AttributeError:
        #             _logger.debug(
        #                 'Section of abstract "%s" with pmid %s has no label.',
        #                 title,
        #                 pmid,
        #             )
        #             continue
        #         try:
        #             cat = self.abstract_labels[label]
        #             correct_label = ABSTRACT_PART[cat]
        #         except KeyError:
        #             _logger.debug("Found unknown abstract label: '%s'.", label)
        #             correct_label = ABSTRACT_PART.invalid
        #         if correct_label is not ABSTRACT_PART.invalid:
        #             categories[correct_label].append(str(section))
        # except KeyError:
        #     _logger.debug(
        #         "Article \"%s\" with PMID %s has no 'Abstract' key...", title, pmid
        #     )
        #     return None
        # result = ["\n".join(categories[cat]) for cat in categories.keys()]
        # lens = [len(i) for i in result]
        abstract = "".join(results)
        normalized_abstract = preprocessing.normalize_abstract(abstract,datapath=self.)
        parsed: spacy.tokens.Doc
        parsed = self.nlp(abstract)

        prev = 0
        # for i, l in enumerate(lens):
        #     end = prev + l
        #     if end != prev:
        #         sp = parsed.char_span(prev, end)
        #         sp._.abstract_part = list(categories.keys())[i]
        #     prev = prev + l
        parsed._.title = title
        parsed._.journal = journal
        parsed._.pmid = pmid
        parsed._.date = date
        _logger.debug("Parsed article into spacy doc.")
        return parsed


class DataManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        sessions = join(self.data_dir, "sessions")
        self.datasets: Dict[str, str]
        self.datasets = {}
        if not exists(sessions):
            os.makedirs(sessions)
        for d in listdir(sessions):
            if os.path.isdir(join(sessions, d)):
                _logger.info("Found dataset: {}".format(d))
                self.datasets[d] = join(sessions, d)

    def add_or_get_dataset(self, name: str, query: str) -> Optional[DataSet]:
        if name in self.datasets.keys():
            ds = self.datasets[name]
            ds_loaded = DataSet(ds)
            assert (
                query == ds_loaded.metadata.query
            ), "error: supplied query differs from existing dataset's query"
            return ds_loaded
        else:
            new_dataset = join(self.data_dir, "sessions", name)
            os.makedirs(new_dataset)
            self.datasets[name] = new_dataset
            return DataSet(new_dataset, query)


if __name__ == "__main__":
    data = DataManager(default_data)
    test_query = """"hypersomnia" [All Fields] AND medline[sb] AND "2009/03/20"[PDat] : "2019/03/17"[PDat] AND "humans"[MeSH Terms]"""
    test_dataset = data.add_or_get_dataset("test_hypersomnia", test_query)

    assert test_dataset is not None
    test_dataset.fetch_ids()
    abs_gen = test_dataset.get_raw_abstracts_generator()
    test_docs = [test_dataset.make_abstract_doc(d) for d in abs_gen]
    print(test_doc)
