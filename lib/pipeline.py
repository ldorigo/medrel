"""Main entry point for the project, where the entire pipeline is defined and executed.
"""
import itertools
import argparse
from typing import Generator
from pathlib import Path
from tqdm.std import tqdm
import lib.pipe_pubmed as pipe_pubmed
import lib.pipe_preprocessing as pipe_preprocessing
import lib.pipe_spacy as pipe_spacy
import lib.grammar_analysis as grammar_analysis
import spacy


def save_to_folder(path: str, query: str):
    total_ids, ids_generator = pipe_pubmed.get_pmids_generator(query=query)

    def inner_gen() -> Generator[str, None, None]:
        for progress, id in tqdm(ids_generator, total=total_ids):
            yield id

    raw_abstracts_generator = pipe_pubmed.get_raw_abstracts_generator(inner_gen())
    ag = pipe_pubmed.get_abstracts_generator(raw_abstracts_generator)
    pipe_pubmed.save_abstracts_to_files(ag, Path(path))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    group = argparser.add_mutually_exclusive_group()
    group.add_argument(
        "-l",
        "--local",
        help="Local directory from which to get abstracts instead of downloading from pubmed.",
    )
    group.add_argument(
        "-q", "--query", help="Query for which to retrieve pubmed abstracts."
    )

    argparser.add_argument(
        "-s",
        "--save_abstracts",
        help="Save raw abstracts to specified folder for further use with the -l option.",
    )
    args = argparser.parse_args()

    # If a query is given, get the abstract generator via pubmed
    if args.query:
        total_ids, ids_generator = pipe_pubmed.get_pmids_generator(query=args.query)

        def inner_gen() -> Generator[str, None, None]:
            for progress, id in tqdm(ids_generator, total=total_ids):
                yield id

        raw_abstracts_generator = pipe_pubmed.get_raw_abstracts_generator(inner_gen())
        ag = pipe_pubmed.get_abstracts_generator(raw_abstracts_generator)
        if args.save_abstracts:
            # Split the iterator to both save to file and continue preprocessing
            ag, ag2 = itertools.tee(ag, 2)

            pipe_pubmed.save_abstracts_to_files(ag2, Path(args.save_abstracts))

    # Otherwise if --local is set, get the generator from a local directory
    elif args.local:
        ag = pipe_pubmed.abstracts_from_files_generator(Path(args.local))
    else:
        raise RuntimeError("Must pass either --query or --local")

    # The rest of the pipeline is common for both options
    preprocessed_generator = pipe_preprocessing.get_preprocessed_abstracts_generator(ag)

    nlp = spacy.load("en_core_sci_md", exclude=["ner"])
    raw_docs_generator = pipe_spacy.get_raw_doc_generator(preprocessed_generator, nlp)
    docs_generator = pipe_spacy.get_extended_doc_generator(raw_docs_generator)
    relevant_sentences_generator = pipe_spacy.get_relevant_sentence_numbers_generator(
        docs_generator
    )
    doc_relations_generator = grammar_analysis.get_relations_generator(
        relevant_sentences_generator
    )
    rels: grammar_analysis.Relations = []
    for doc, relations_dict in itertools.islice(doc_relations_generator, 1000):
        for i, relations in relations_dict.items():
            if relations:
                # rels += relations
                print(relations)

    # with open("relations_sample.json", "w+") as f:
    # json.dump(rels, f, cls=TokenEncoder)
