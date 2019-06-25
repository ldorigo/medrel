"""Manipulate abstracts in pandas dataframe"""
import shutil
from os.path import exists, join
from typing import Dict

import numpy as np
import pandas
import re
import wget
from abbreviations import schwartz_hearst

import constants
from utils import logger


def load_unicode_mappings(datapath: str) -> Dict[int, str]:
    """
    Load mappings between unicode and ascii characters.
    datapath: path to data directory
    """
    mappings = {}
    filename = "entities.dat"

    if not exists(join(datapath, filename)):
        url = "https://structuredabstracts.nlm.nih.gov/Downloads/Structured-Abstracts-Labels-102615.txt"
        logger.debug(
            "File with unicode-to-ascii mappings not found. Downloading from %s.", url
        )
        filename = wget.download(url)
        shutil.move(filename, join(datapath, filename))
        logger.debug("Finished downloading file.")

    with open(join(datapath, filename), "r") as f:
        for line in f:
            if line[0] == "#":
                pass
            else:
                uni, asc = line.split("\t")
                asc = asc.strip()
                mappings[int(uni, 16)] = asc
    return mappings


def unicode2ascii(mappings: Dict[int, str], text: str) -> str:
    """
    transform all unicode characters in a string into ascii-equivalents
    """

    text_list = list(text)
    for index, char in enumerate(text_list):
        if ord(char) > 127:
            # Non-ascii character.
            try:
                text_list[index] = mappings[ord(char)]
            except KeyError:
                text_list[index] = ""
    return "".join(text_list)


def get_full_text(article):
    """
    Get a pretty-printed version of the article
    """
    res_arr = []
    for section in constants.BODY_COLUMNS:
        print(article[section])
        if article[section] and str(article[section]) != "nan":
            res_arr.append(str(article[section]))

    return "Article: " + str(article["title"]) + "\n\n" + "\n".join(res_arr)


def get_relevant_text(abstract_row: pandas.Series) -> str:
    relevant_sections = [
        abstract_row.title,
        abstract_row.background,
        abstract_row.results,
        abstract_row.conclusions,
    ]
    return " ".join(relevant_sections)


def get_abbreviations(text):
    """
    Get list of abbreviations in sentence
    """
    pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=text)
    return pairs


def get_abstract_abbreviations(art):
    """
    Get all abbreviations defined in abstract. Return a dictionnary of mappings from abbreviation to expanded form
    """
    return get_abbreviations(" ".join(art.iloc[0:6]))


def replace_abbreviations(txt, abbs):
    """
    Replace all occurrences of the abbreviations in the given text.
    
    >>> replace_abbreviations("ABS has been linked", {"ABS": "Anti Break System"})
    'Anti Break System has been linked'
    
    >>> replace_abbreviations("ABSOLUTE VALUE", {"ABS": "Anti Break System"})
    'ABSOLUTE VALUE'
    """
    res = txt
    for abb in abbs:
        patt = "([^a-zA-Z]|^)" + re.escape(abb) + "([^a-zA-Z])"
        rep = r"\1 {}\2".format(abbs[abb])
        res = re.sub(patt, rep, res)
    return res


def replace_abstract_abbreviations(art):
    abbs = get_abstract_abbreviations(art)
    return art.iloc[0:6].map(lambda a: replace_abbreviations(a, abbs))


def remove_brackets(text: str) -> str:
    types = [("(", ")"), ("[", "]"), ("{", "}")]
    res = []
    has_angle = False
    for t in types:
        bracket_level = 0
        for char in text:
            if char == t[0]:
                bracket_level += 1

            if not bracket_level:
                res.append(char)

            if char == t[1]:
                bracket_level -= 1
            if char == "<":
                has_angle = True

        text = "".join(res)
        res = []
    # Since < and > can also be used as comparison operators, they have to be handled differently
    if has_angle:
        text = remove_angles(text)
    return normalize_whitespace(text)


def remove_angles(text: str) -> str:
    """Remove angled brackets from text. Assuming there's no nested brackets.
    
    Returns:
        str: [description]

    >>> remove_angles("hello <there> I  </sup > \\n need to <3 talk \\n  > to you <3>!")

    """
    m = re.search("<[^><\\n]*>", text)
    while m:
        start, end = m.span()
        text = text[:start] + text[end:]
        m = re.search("<[^><\\n]*>", text)
    return text


def normalize_whitespace(text: str) -> str:
    res = []

    count = 0
    for char in text:
        if char == " ":
            if count == 0:
                res.append(char)
            count += 1
        else:
            res.append(char)
            count = 0
    return "".join(res).strip()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
