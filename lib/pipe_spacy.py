"""Convert from entrez abstracts into spacy doc, while saving relevant information
"""
from pathlib import Path
from typing import Generator, List, Tuple, Iterator
import lib.pipe_pubmed as pipe_pubmed
import spacy
from tqdm import tqdm
import time

from spacy.matcher import Matcher
from lib.constants import (
    PATTERNS_NEUTRAL_RELATIONS,
    PATTERNS_NEGATION,
    PATTERNS_IGNORED_WORDS,
)


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


def get_raw_doc_generator(
    abstract_generator: Iterator[Tuple[str, pipe_pubmed.AbstractMetadata]],
    nlp: spacy.Language,
) -> Generator[Tuple[spacy.tokens.Doc, pipe_pubmed.AbstractMetadata], None, None]:
    yield from nlp.pipe(abstract_generator, as_tuples=True, n_process=-1)


def get_extended_doc_generator(
    raw_doc_generator: Iterator[Tuple[str, pipe_pubmed.AbstractMetadata]]
) -> Generator[spacy.tokens.Doc]:
    for raw_doc in raw_doc_generator:
        raw_doc._.title = title
        raw_doc._.journal = journal
        raw_doc._.pmid = pmid
        raw_doc._.date = date
        yield raw_doc


# def get_relevant_sentence_numbers_generator(
#     extended_doc_generator: Iterator[spacy.tokens.Doc],
# ) -> Generator[Tuple[spacy.tokens.Doc, List[int]]]:
#     for doc in extended_doc_generator:

#         matcher_relations = Matcher(nlp.vocab)
#         matcher_negations = Matcher(nlp.vocab)
#         matcher_relations.add("relation", None, *PATTERNS_NEUTRAL_RELATIONS)
#         matcher_negations.add(
#             "ignore", None, *PATTERNS_NEGATION, *PATTERNS_IGNORED_WORDS
#         )
#         for sent, pmid in class_sent_gen:
#             nmatches = matcher_negations(sent)
#             if nmatches:
#                 continue
#             rmatches = matcher_relations(sent)
#             if rmatches:
#                 yield (sent, pmid)


if __name__ == "__main__":

    test_query = """"hypersomnia" [All Fields] AND medline[sb] AND "2009/03/20"[PDat] : "2019/03/17"[PDat] AND "humans"[MeSH Terms]"""

    # nlp = spacy.load("en_core_sci_md")
    nlp = spacy.load("en_core_sci_md", exclude=["ner"])
    # total_ids, ids_generator = pipe_pubmed.get_pmids_generator(query=test_query)

    # def inner_gen() -> Generator[str, None, None]:
    #     for progress, id in tqdm(ids_generator, total=total_ids):
    #         yield id

    # raw_abstracts_generator = pipe_pubmed.get_raw_abstracts_generator(inner_gen())
    # abstracts_generator = pipe_pubmed.get_abstracts_generator(raw_abstracts_generator)
    abstracts_generator = pipe_pubmed.abstracts_from_files_generator(
        Path("../data/test_data")
    )
    abstracts = list(abstracts_generator)
    start = time.time()
    raw_doc_generator = get_raw_doc_generator(abstracts, nlp=nlp)
    i = 0
    for abstract in raw_doc_generator:
        # print(abstract)
        i += 1

    print(f"Elapsed time: {time.time() - start}")
    print(f"i={i}")
# %%
