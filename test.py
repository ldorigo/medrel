import unittest
import spacy
from lib.grammar_analysis import parse_sentence, pretty_print_relation


nlp = spacy.load("en_core_sci_md")


class TestParser(unittest.TestCase):
    def test_passive(self):
        sent_1 = "By meta-analysis, we found A1166C polymorphism was associated with decreased risk for breast cancer in Caucasian population in an additive model . "
        expected_result = ["[[A1166C, polymorphism] (~)] <-> [[breast, cancer] (↓)]"]

        ns = nlp(sent_1)
        relations = parse_sentence(ns, nlp)
        pp = [pretty_print_relation(rel) for rel in relations]
        self.assertSequenceEqual(expected_result, pp)


if __name__ == "__main__":
    unittest.main()
