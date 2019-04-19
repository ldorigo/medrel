"""
Functions to remove unicode characters
"""

from typing import Dict
from os import path
import logging
import shutil
import wget
logging.basicConfig(level=logging.DEBUG)


def load_unicode_mappings(datapath: str) -> Dict[int, str]:
    """
    Load mappings between unicode and ascii characters.
    datapath: path to data directory
    """
    mappings = {}
    filename = "entities.dat"

    if not path.exists(path.join(datapath, filename)):
        url = "https://structuredabstracts.nlm.nih.gov/Downloads/Structured-Abstracts-Labels-102615.txt"
        logging.debug(
            "File with unicode-to-ascii mappings not found. Downloading from %s.", url)
        filename = wget.download(url)
        shutil.move(filename, path.join(datapath, filename))
        logging.debug("Finished downloading file.")

    with open(path.join(datapath, filename), "r") as f:
        for line in f:
            if line[0] == '#':
                pass
            else:
                uni, asc = line.split('\t')
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
                text_list[index] = ''
    return "".join(text_list)


def get_full_text(article):
    """
    Get a pretty-printed version of the article
    """
    sections = ["background", "objective", "methods", "results", "conclusions"]
    res_arr = []
    for section in sections:
        print(article[section])
        if article[section] and str(article[section]) != "nan":
            res_arr.append(str(article[section]))

    return "Article: " + str(article['title']) + "\n\n" + "\n".join(res_arr)
