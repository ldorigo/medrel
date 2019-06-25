import unittest

from grammar_analysis import get_nlp, parse_sentence

from testcases import testcases

class TestParses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = get_nlp()

    def test_upper(self):

        def test_sent(sent):
            ns = self.nlp(sent)
            return parse_sentence(ns, self.nlp)

        for i, testcase in enumerate(testcases):
            print("Running testcase {}: {}".format(i,testcase))
            sent = testcase
            exp = testcases[testcase]
            res = test_sent(sent)
            self.assertEqual( repr(res), exp )

if __name__ == "__main__":
    unittest.main()

