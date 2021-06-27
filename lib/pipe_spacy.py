"""Convert from entrez abstracts into spacy doc, while saving relevant information
"""
from pathlib import Path
from typing import Generator, Iterable, List, Tuple, Iterator
import lib.pipe_pubmed as pipe_pubmed
import lib.pipe_preprocessing as pipe_preprocessing
import spacy
from spacy import tokens
from scispacy.abbreviation import AbbreviationDetector
import time

from spacy.matcher import Matcher
from lib.constants import (
    PATTERNS_NEUTRAL_RELATIONS,
    PATTERNS_NEGATION,
    PATTERNS_IGNORED_WORDS,
)


def set_extensions():
    """Add relevant extensions to spacy docs and spans"""
    if not tokens.Doc.has_extension("journal"):
        tokens.Doc.set_extension("journal", default="")
    if not tokens.Doc.has_extension("title"):
        tokens.Doc.set_extension("title", default="")
    if not tokens.Doc.has_extension("pmid"):
        tokens.Doc.set_extension("pmid", default=0)
    if not tokens.Doc.has_extension("date"):
        tokens.Doc.set_extension("date", default=0)


set_extensions()


def get_raw_doc_generator(
    abstract_generator: Iterable[Tuple[str, pipe_pubmed.AbstractMetadata]],
    nlp: spacy.Language,
) -> Generator[Tuple[tokens.Doc, pipe_pubmed.AbstractMetadata], None, None]:
    """Get tuples of (abstract_text, metadata) and yield tuples of (Doc, metadata) where Doc is the spacy doc corresponding to the original text.

    Args:
        abstract_generator (Iterator[Tuple[str, pipe_pubmed.AbstractMetadata]]): iterator of tuples where the first element is abstract text and second element is a
        dict with abstract metadata
        nlp (spacy.Language): Spacy Language object

    Yields:
        Generator[Tuple[spacy.tokens.Doc, pipe_pubmed.AbstractMetadata], None, None]: Generator that yields tuples of
            1. spacy docs containing the abstract text
            2. the same metadata as in the tuples received as argument
    """
    nlp.add_pipe("abbreviation_detector")
    yield from nlp.pipe(abstract_generator, as_tuples=True, n_process=1)


def get_extended_doc_generator(
    raw_doc_generator: Iterable[Tuple[tokens.Doc, pipe_pubmed.AbstractMetadata]]
) -> Generator[tokens.Doc, None, None]:
    """Get tuples of spacy docs and metadata and merge them by saving the metadata as Doc underscore (._.) extensions.
    Note: These are the basic objects that need to be stored on a large scale.

    Args:
        raw_doc_generator (Iterator[Tuple[str, pipe_pubmed.AbstractMetadata]]): tuples of spacy docs and metadata as returned by get_raw_doc_generator

    Yields:
        Generator[spacy.tokens.Doc]: spacy docs with article metadata saved as underscore extensions.
    """
    for raw_doc, context in raw_doc_generator:
        raw_doc._.title = context["title"]
        raw_doc._.journal = context["journal"]
        raw_doc._.pmid = context["pmid"]
        raw_doc._.date = context["date"]
        yield raw_doc


def get_relevant_sentence_numbers_generator(
    extended_doc_generator: Iterator[tokens.Doc],
) -> Generator[Tuple[tokens.Doc, List[int]], None, None]:
    """Iterate through the docs passed as input and yield the same docs together with the list of indices of the sentences in those docs that contain relations.

    Args:
        extended_doc_generator (Iterator[tokens.Doc]): Iterator that yields Doc objects

    Yields:
        Generator[Tuple[tokens.Doc, List[int]], None, None]: Generator that yields tuples such that the first element is the original Doc, and the second element is the list of indices of the sentences of that doc that contain a relation indicator.
    """
    for doc in extended_doc_generator:

        matcher_relations = Matcher(doc.vocab)
        matcher_negations = Matcher(doc.vocab)
        matcher_relations.add("relation", patterns=PATTERNS_NEUTRAL_RELATIONS)
        matcher_negations.add(
            "ignore", patterns=PATTERNS_NEGATION + PATTERNS_IGNORED_WORDS
        )
        relevant_sentences: List[int] = []
        for i, sent in enumerate(doc.sents):
            nmatches = matcher_negations(sent)
            if nmatches:
                continue
            rmatches = matcher_relations(sent)
            if rmatches:
                relevant_sentences.append(i)
        yield (doc, relevant_sentences)


if __name__ == "__main__":

    test_query = """"hypersomnia" [All Fields] AND medline[sb] AND "2009/03/20"[PDat] : "2019/03/17"[PDat] AND "humans"[MeSH Terms]"""

    nlp = spacy.load("en_core_sci_md")
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
    abstracts_preprocessed = pipe_preprocessing.get_preprocessed_abstracts_generator(
        abstracts
    )
    start = time.time()
    raw_doc_generator = get_raw_doc_generator(abstracts_preprocessed, nlp=nlp)
    extended_doc_generator = get_extended_doc_generator(raw_doc_generator)
    sents_gen = get_relevant_sentence_numbers_generator(extended_doc_generator)
    for doc, sent_idxs in sents_gen:
        if sent_idxs:
            print(f"{doc._.title}\n")
            print(f"Found the following abbreviations:")
            for abrv in doc._.abbreviations:
                print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
            print(f"Relevant Sentences:")
            for sent in sent_idxs:
                print(f"\t {list(doc.sents)[sent]}")

            print("\n\n")
    print(f"Elapsed time: {time.time() - start}")
