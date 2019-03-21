import logging
from utilities import Abstract_Part, open_local_article_set, load_structured_labels
from Bio import Entrez
import pickle
import json

logging.basicConfig(level=logging.DEBUG)


class Abstract:
    def __init__(self, article_dict):
        self.text = {i: [] for i in Abstract_Part}
        self.pmid = str(article_dict["MedlineCitation"]["PMID"])
        assert self.pmid != ""

        article = article_dict["MedlineCitation"]["Article"]
        self.date = dict(article_dict["MedlineCitation"]["DateCompleted"])

        lang = article["Language"][0]
        if lang != "eng":
            # logging.debug("Article Language was not english ({})".format(lang))
            raise ValueError("Article is not in english")
        self.title = str(article["ArticleTitle"])
        if len(self.title) < 5:
            logging.error("Error: Something went wrong (article title is too short)")
            raise ValueError("Article has misformed title")
        self.journal = str(article["Journal"]["Title"])
        assert len(self.journal) != 0, "Error: Article journal's name is not defined"
        for section in article["Abstract"]["AbstractText"]:
            try:
                label = section.attributes["Label"].lower().strip()
            except AttributeError:
                logging.error("Section has no label")
                continue
            try:
                st_label = category_mappings[label]
            except KeyError:
                logging.error("Found unknown abstract label: '{}'.".format(label))
                st_label = Abstract_Part(6)

            if st_label != Abstract_Part(6):
                self.text[st_label].append(str(section))
        for cat in self.text:
            self.text[cat] = " ".join(self.text[cat])

    def __str__(self):
        pass
        # return "Abstract from article: {}, published on: {}/{}/{}, in {}.".format(self.title,self.date[0])


articles = open_local_article_set("data/test_run")
category_mappings = load_structured_labels(
    "data/Structured-Abstracts-Labels-102615.txt"
)

i = 0
results = list()
for article in articles():
    try:
        results.append(Abstract(article))
        i += 1
    except ValueError:
        pass

    if i % 100 == 0:
        print("Processing abstract {}".format(i))

with open("./data/article_list_obesity.pickle", "wb+") as f:
    try:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    except pickle.PicklingError:
        logging.error("Could not pickle.")
