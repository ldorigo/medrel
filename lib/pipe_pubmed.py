"""Module handling the retrieval of raw abstract texts from pubmed.
"""

from pathlib import Path
from typing import (
    Generator,
    Iterator,
    TypedDict,
    Optional,
    Tuple,
    Set,
    Dict,
    Union,
)
import json
import itertools
from Bio import Entrez
from Bio.Entrez import Parser
from tqdm import tqdm

from dateutil import parser as dateparser

Entrez.email = "ludor19@student.sdu.dk"

EntrezElement = Union[
    Parser.ListElement,
    Parser.DictionaryElement,
    Parser.NoneElement,
    Parser.IntegerElement,
]


class AbstractMetadata(TypedDict):
    title: str
    journal: str
    pmid: str
    date: float  # unix timestamp


def date_to_timestamp(date_dict: Dict[str, str]) -> float:
    """Convert from entrez date format to a normal timestamp

    Args:
        date_dict (Dict[str, str]): [description]

    Returns:
        float: [description]
    """
    try:
        datestr = date_dict["Month"] + "." + date_dict["Day"] + "." + date_dict["Year"]
        date = dateparser.parse(datestr).timestamp()
    except KeyError:
        date = 0
    return date


def get_pmids_generator(
    query: str,
    ignore: Optional[Set[str]] = None,
    batch_size: int = 10000,
    max_amount: int = 0,
) -> Tuple[int, Generator[Tuple[int, str], None, int]]:

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
    ids_handle.close()

    def inner_generator() -> Generator[Tuple[int, str], None, int]:
        # How many ids were fetched
        downloaded_ids_count = 0
        # How many ids were yielded (i.e. the above minus the ones in `ignore`)
        yielded_ids_count = 0
        # How many ids were processed (for smooth progress information)
        processed_ids_count = 0
        # Which batch we're at:
        ids_iteration_count = 0

        while downloaded_ids_count < amount:
            if max_amount and yielded_ids_count >= max_amount:
                return yielded_ids_count
            # _logger.debug("Fetching id list...")
            # We get the next batch of ids:
            next_ids_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=batch_size,
                retstart=batch_size * ids_iteration_count,
            )

            next_ids_record: Optional[EntrezElement] = Entrez.read(next_ids_handle)
            assert next_ids_record and isinstance(
                next_ids_record, Parser.DictionaryElement
            )

            next_ids_handle.close()
            next_ids: Set[str] = set(next_ids_record["IdList"])
            downloaded_ids_count += batch_size
            ids_iteration_count += 1

            for _id in next_ids:
                processed_ids_count += 1
                if not ignore or _id not in ignore:
                    yielded_ids_count += 1
                    yield (processed_ids_count, _id)
        return yielded_ids_count

    return (amount, inner_generator())


def get_raw_abstracts_generator(
    ids: Iterator[str], batch_size: int = 100
) -> Iterator[Parser.DictionaryElement]:

    while next_ids := list(itertools.islice(ids, batch_size)):
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


def get_abstracts_generator(
    raw_abstracts_generator: Iterator[Parser.DictionaryElement],
) -> Generator[Tuple[str, AbstractMetadata], None, None]:
    for abstract_dict in raw_abstracts_generator:
        pmid: str = str(abstract_dict["MedlineCitation"]["PMID"])
        abstract = abstract_dict["MedlineCitation"]["Article"]
        try:
            date: float = date_to_timestamp(
                dict(abstract_dict["MedlineCitation"]["DateCompleted"])
            )
        except KeyError:
            date = 0
        # If the abstract is not in english don't consider it
        lang: str = abstract["Language"][0]
        if lang != "eng":
            continue

        title = str(abstract["ArticleTitle"])
        if len(title) < 5:
            continue
        journal = str(abstract["Journal"]["Title"])
        if not journal:
            continue
        try:
            results = [str(section) for section in abstract["Abstract"]["AbstractText"]]
        except KeyError:
            continue
        metadata: AbstractMetadata = {
            "pmid": pmid,
            "title": title,
            "date": date,
            "journal": journal,
        }
        results_text = "\n".join(results)
        yield (results_text, metadata)


def save_abstracts_to_files(
    abstracts_generator: Iterator[Tuple[str, AbstractMetadata]], data_dir: Path
):
    assert data_dir.exists() and data_dir.is_dir()
    for abstract in abstracts_generator:
        with open(data_dir / f"{abstract[1]['pmid']}.json", "w+") as f:
            json.dump(abstract, f)


def abstracts_from_files_generator(
    folder_path: Path,
) -> Generator[Tuple[int, str], None, None]:
    i = 0
    while i < 2000:
        for file in folder_path.glob("*.json"):
            with open(file, "r") as f:
                abstract = json.load(f)
                i += 1
                yield abstract


if __name__ == "__main__":
    test_query = """"hypersomnia" [All Fields] AND medline[sb] AND "2009/03/20"[PDat] : "2019/03/17"[PDat] AND "humans"[MeSH Terms]"""

    total_ids, ids_generator = get_pmids_generator(query=test_query)

    def inner_gen() -> Generator[str, None, None]:
        for progress, id in tqdm(ids_generator, total=total_ids):
            yield id

    raw_abstracts_generator = get_raw_abstracts_generator(inner_gen())
    ag = get_abstracts_generator(raw_abstracts_generator)
    data_path = Path("../data/test_data")
    # data_path.mkdir()
    save_abstracts_to_files(ag, data_path)
