import spacy
import pickle

from utils import logger


def set_extensions():
    """Add relevant extensions to spacy docs and spans
    
    """
    if not spacy.tokens.Doc.has_extension("journal"):
        spacy.tokens.Doc.set_extension("journal", default="")
    if not spacy.tokens.Doc.has_extension("pmid"):
        spacy.tokens.Doc.set_extension("pmid", default=0)
    if not spacy.tokens.Doc.has_extension("date"):
        spacy.tokens.Doc.set_extension("date", default=0)
    if not spacy.tokens.Span.has_extension("CUI"):
        spacy.tokens.Span.set_extension("CUI", default='UNKNOWN')



def ner_on_abstract(abstract_row,matcher, nlp):

    all_text = " ".join(abstract_row[0:6])
    relevant_sections = [abstract_row.title, abstract_row.background,
                         abstract_row.results, abstract_row.conclusions]
    relevant_text = " ".join(relevant_sections)

    logger.debug("Processing abstract: {}".format(relevant_text))

    # Tokenization, POS tagging and all the schmuck with vanilla spacy.
    doc = nlp(relevant_text)

    doc._.journal = abstract_row.journal
    doc._.pmid = abstract_row.pmid
    doc._.date = abstract_row.date
    doc_with_ents = matcher.match_spacy(doc)
    return doc_with_ents


def ner_on_text(text,matcher, nlp):

    logger.debug("Processing text: {}".format(text))

    # Tokenization, POS tagging and all the schmuck with vanilla spacy.
    doc = nlp(text)

#     doc._.journal = abstract_row.journal
#     doc._.pmid = abstract_row.pmid
#     doc._.date = abstract_row.date
    doc_with_ents = matcher.match_spacy(doc)
    return doc_with_ents


