"""Module handling the retrieval of raw abstract texts from pubmed.
"""

from typing import Iterator, List, Optional, Set, Any, Dict, Union
import itertools
from Bio import Entrez
from Bio.Entrez import Parser

Entrez.email = "ludor19@student.sdu.dk"

EntrezElement = Union[
    Parser.ListElement,
    Parser.DictionaryElement,
    Parser.NoneElement,
    Parser.IntegerElement,
]


def get_pmids_generator(
    query: str,
    ignore: Optional[Set[str]] = None,
    batch_size: int = 10000,
    max_amount: int = 0,
) -> Iterator[str]:

    # Get the amount of results
    ids_handle = Entrez.egquery(term=query)
    record: Optional[EntrezElement]
    record = Entrez.read(ids_handle)
    assert record and isinstance(record, Parser.DictionaryElement)
    amount = 0
    for row in record["eGQueryResult"]:
        if row["DbName"] == "pubmed":
            amount = int(row["Count"])
    assert amount

    # How many ids were fetched
    downloaded_ids_count = 0
    # How many ids were yielded (i.e. the above minus the ones in `ignore`)
    yielded_ids_count = 0
    # Which batch we're at:
    ids_iteration_count = 0

    while downloaded_ids_count < amount:
        if yielded_ids_count >= max_amount:
            return
        # _logger.debug("Fetching id list...")
        # We get the next batch of ids:
        next_ids_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=batch_size,
            retstart=batch_size * ids_iteration_count,
        )

        next_ids_record: Optional[EntrezElement] = Entrez.read(next_ids_handle)
        assert next_ids_record and isinstance(next_ids_record, Parser.DictionaryElement)

        next_ids_handle.close()
        next_ids: Set[str] = set(next_ids_record["IdList"])
        downloaded_ids_count += batch_size
        ids_iteration_count += 1

        for id in next_ids:
            if id not in ignore:
                yielded_ids_count += 1
                yield id


def get_raw_abstracts_generator(
    ids: Iterator[str], batch_size: int = 100
) -> Iterator[Parser.DictionaryElement]:
    next_ids = list(itertools.islice(ids, batch_size))
    next_abstracts_handle = Entrez.efetch(db="pubmed", id=next_ids, retmode="xml")
    next_abstracts_record: Optional[EntrezElement] = Entrez.read(
        next_abstracts_handle, validate=False
    )
    next_abstracts_handle.close()
    assert next_abstracts_record and isinstance(
        next_abstracts_record, Parser.DictionaryElement
    )
    for item in next_abstracts_record["PubmedArticle"]:
        yield item