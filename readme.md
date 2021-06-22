# Medrel: rule-based Biomedical Relations extraction

## Project structure

Library is written as a functional pipeline using generators for lazy evaluation. The main components are:

- `pipe_pubmed`: to retrieve raw abstracts from pubmed and extract text and metadata
- `pipe_preprocessing`: to normalize the texts (convert from unicode to ascii, remove brackets, normalize whitespace, etc.)
- `pipe_spacy`: to parse the abstracts into spaCy docs and do POS tagging (and also abbreviation resolution). Also has a pipeline component to find candidate sentences for relation extraction.
- `grammar_analysis`: the actual relation extraction, not yet up to date or working.

Other files/folders:

- bak: old stuff that will eventually be deleted, kept for now to be safe.
- notebooks: old notebooks from bachelor projects - might contain a few example but are outdated, will be removed soon.
