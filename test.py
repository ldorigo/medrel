import unittest
import spacy
from lib.grammar_analysis import parse_sentence, pretty_print_relation


nlp = spacy.load("en_core_sci_md")


class TestParser(unittest.TestCase):
    def test_passive_association(self):
        sents_dict = {
            "By meta-analysis, we found A1166C polymorphism was associated with decreased risk for breast cancer in Caucasian population in an additive model . ": [
                "[[A1166C, polymorphism] (~)] <-> [[breast, cancer] (↓)]"
            ],
            "The use of HAART was associated with a significant increase in both physical and mental aspects of the HRQOL over a 12-month period in this urban African population.": [
                "[[HAART] (~)] <-> [[HRQOL] (↑)]"
            ],
            "CCL2 responses to Mycobacterium tuberculosis are associated with disease severity in tuberculosis.": [
                "[[CCL2, responses] (~)] <-> [[tuberculosis] (~)]",
                "[[CCL2, responses, to, Mycobacterium, tuberculosis] (~)] <-> [[tuberculosis] (~)]",
            ],
            "Gestational diabetes mellitus is associated with higher RT during pregnancy compared with non-GDM.": [
                "[[Gestational, diabetes, mellitus] (~)] <-> [[RT] (↑)]",
                "[[diabetes, mellitus] (~)] <-> [[RT] (↑)]",
            ],
        }
        for sentence in sents_dict:

            ns = nlp(sentence)
            relations = parse_sentence(ns, nlp)
            pp = [pretty_print_relation(rel) for rel in relations]
            self.assertSequenceEqual(sents_dict[sentence], pp)


if __name__ == "__main__":
    unittest.main()
