""" Various utilities used through the project"""

import logging
import re
from os.path import join
from typing import Dict, List, Optional, Set, Tuple

import ipywidgets as widgets
import numpy as np
import pandas
import spacy
from tqdm.auto import tqdm
from dateutil import parser as dateparser

from defaults import (
    default_session,
    default_data,
    ,
)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger("notebook")
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info("Created logger with name: {}".format(name))
    return logger


_logger = get_logger("utils")


DATALOCATION_WIDGET = widgets.Text(value=default_data, description="Data location:")
SESSIONLOCATION_WIDGET = widgets.Text(
    value=default_session, description="Session location:"
)
QUICKUMLS_LOCATION_WIDGET = widgets.Text(
    value=default_quickumls_loc, description="QuickUMLS path:"
)
MRCONSO_PATH = default_mrconso


def save_df(df: pandas.DataFrame, name: str) -> None:
    """Save pandas dataframe to csv file with given name.

    Args:
        df (pandas.DataFrame): dataframe to be saved
        name (str): filename
    """
    df.to_csv(join(get_session_dir(), name), index=False)


def load_if_not_present(
    variable: Optional[pandas.DataFrame], name: str
) -> pandas.DataFrame:
    """Check if given variable currently contains a dataframe, and if not, load it from provided file.

    Args:
        variable (Optional[pandas.DataFrame]): Variable that could be None or could contain a df
        name (str): file in which the saved df can be found.

    Returns:
        pandas.DataFrame
    """
    if variable is None:
        path = join(get_session_dir(), name)
        variable = pandas.read_csv(path)
        _logger.debug("Loaded {} from csv.".format(name))
        return variable.replace(np.nan, "", regex=True)
    else:
        _logger.debug("Reusing previously defined df.")
        return variable


def get_data_dir() -> str:
    """Get currently defined data directory (from ipython widgets)

    Returns:
        str: path to the directory containing all data.
    """
    return DATALOCATION_WIDGET.value


def get_session_dir() -> str:
    """Get currently defined session directory (from ipython widgets)

    Returns:
        str: path to the directory containing session data.
    """
    return join(DATALOCATION_WIDGET.value, "sessions", SESSIONLOCATION_WIDGET.value)


def extract_UMLS_types(su_file: str) -> Tuple[List[str], Dict[str, str]]:
    """Get both a list of nicely-formatted strings and a dictionnary of UMLS semantic types identifiers.
    Mostly used to improve QuickUMLS.

    Args:
        su_file (str): path to file containing UMLS semantic type definitions.

    Returns:
        Tuple[List[str], Dict[str, str]]: A tuple containing both a list of semantic type identifiers (used in constants.py)
        and a dict mapping betwee semnatic type identifier and canonical name.
    """
    contents = open(su_file, "r").read()
    concepts = re.split("\n\n", contents)
    uireg = re.compile("UI: (.*)")
    namereg = re.compile("STY: (.*)")
    exreg = re.compile("EX: (.*)")
    lines: List[str]
    lines = []
    concepts_dict: Dict[str, str]
    concepts_dict = {}
    for concept in concepts:
        uires = uireg.search(concept)
        nameres = namereg.search(concept)
        exres = exreg.search(concept)
        if not uires or not nameres or not exres:
            break  # The end of the file contains relations, which I'm not interested in
        else:
            ui = uires.groups()[0]
            name = nameres.groups()[0]
            ex = exres.groups()[0]
            lines.append("'{}', # {}, ex.: {}".format(ui, name, ex))
            concepts_dict[ui] = name
    return (lines, concepts_dict)


def get_unique_cuis(abstract_list: List[spacy.tokens.Doc]) -> List[str]:
    """Get a list of unique CUIs detected by quickumls on a list of abstracts.

    Args:
        abstract_list (List[spacy.tokens.Doc]): list of Spacy DOCs on which NER was already performed.

    Returns:
        List[str]: list of encountered CUIs.
    """
    unique_cuis: Set[str]
    unique_cuis = set()

    for abstract in tqdm(abstract_list):
        cuis = set(map(lambda x: x._.CUI, abstract.ents))
        unique_cuis.update(cuis)

    unique_cuis_list = sorted(list(unique_cuis))
    print("Identified {} unique CUIs".format(len(unique_cuis)))
    return unique_cuis_list


def cuis_to_dict(cuis: List[str]) -> Dict[str, int]:
    return dict([(i, e) for e, i in enumerate(cuis)])


def get_all_cui_dict(path: str = None) -> Dict[str, str]:
    """Get a dictionnary mapping CUIS to standard names for all CUIs in the UMLS. CAUTION: requires 2+ Gb of RAM !!

    Args:
        path (str, optional): Path to the MRCONSO.pipe file part of a UMLS distribution that contains the mappings. Defaults to None (for standard path on the maintainer's machine).

    Returns:
        Dict[str, str]: dictionnary of mappings from CUI to standard name.
    """
    if path is None:
        path = MRCONSO_PATH
    with open(path, "r") as cuis_def_file:
        cuis_names: Dict[str, str]
        cuis_names = dict()
        for line in tqdm(cuis_def_file, total=10304539):
            els: List[str]
            els = line.split("|")
            cuis_names[els[0]] = els[14]
    return cuis_names


def get_cui_name_dict(unique_cuis: List[str], path: str = None) -> Dict[str, str]:
    """Get a dictionnary mapping CUIS to standard name for only the CUI encountered in the processed abstracts.

    Args:
        unique_cuis (List[str]): list of the unique CUIs that were encountered in the currently-being-processed abstracts.
        path (str, optional): Path to the MRCONSO.pipe file part of a UMLS distribution that contains the mappings. Defaults to None (for standard path on the maintainer's machine).

    Returns:
        Dict[str, str]: dictionnary of mappings from CUI to standard name.
    """
    if not path:
        path = MRCONSO_PATH
    cuis_def_file = open(path, "r")
    cuis_names = dict()
    for line in tqdm(cuis_def_file, total=10304539):
        els: List[str]
        els = line.split("|")
        cuis_names[els[0]] = els[14]
    print("Loaded {} CUI names.".format(len(cuis_names)))
    cuis_relevant_names = dict()
    for cui in unique_cuis:
        cuis_relevant_names[cui] = cuis_names[cui]
    cuis_def_file.close()
    return cuis_relevant_names


def get_abstract_relevant_text(abstract_row: pandas.Series) -> str:
    """Get the relevant sections out of a pandas series containing a full abstract

    Args:
        abstract_row (pandas.Series)

    Returns:
        str: The full relevant text
    """
    relevant_sections = [
        abstract_row.title,
        abstract_row.background,
        abstract_row.results,
        abstract_row.conclusions,
    ]
    return " ".join(relevant_sections)


def date_to_timestamp(date_dict: Dict[str, str]) -> float:
    try:
        datestr = date_dict["Month"] + "." + date_dict["Day"] + "." + date_dict["Year"]
        date = dateparser.parse(datestr).timestamp()
    except KeyError:
        date = 0
    return date
