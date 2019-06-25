from typing import Generator, Tuple

import pandas
from tqdm.auto import tqdm
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher

from abstract_manipulations import get_relevant_text

from constants import (
    PATTERNS_NEUTRAL_RELATIONS,
    PATTERNS_NEGATION,
    PATTERNS_IGNORED_WORDS
)

def get_abstract_generator(df: pandas.DataFrame) -> Generator[Tuple[str,int], None, None]:
    for line in tqdm(df.iterrows(), total=len(df)):
        txt = get_relevant_text(line[1])
        yield (txt, line[1]['pmid'])


def get_sentence_generator(
    abs_doc_generator: Generator[Doc, None, None]
) -> Generator[Doc, None, None]:
    for abs_doc,pmid in abs_doc_generator:
        sent: Span
        for sent in abs_doc.sents:
            yield (sent.as_doc(),pmid)


def get_relevant_sentences_generator(class_sent_gen, nlp):
    matcher_relations = Matcher(nlp.vocab)
    matcher_negations = Matcher(nlp.vocab)
    matcher_relations.add("relation", None, *PATTERNS_NEUTRAL_RELATIONS)
    matcher_negations.add("ignore", None, *PATTERNS_NEGATION, *PATTERNS_IGNORED_WORDS)
    for sent,pmid in class_sent_gen:
        nmatches = matcher_negations(sent)
        if nmatches:
            continue
        rmatches = matcher_relations(sent)
        if rmatches:
            yield(sent, pmid)
