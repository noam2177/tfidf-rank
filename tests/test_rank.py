import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank import cosine, idf_map, jaccard, rank, tfidf, tokenize


class RankTests(unittest.TestCase):
    def test_hebrew_letters_stay_one_token(self) -> None:
        self.assertEqual(tokenize("החתול, ישן."), ["החתול", "ישן"])

    def test_prefix_is_not_stripped(self) -> None:
        rows = rank("חתול", ["החתול ישן", "כלב רץ"])
        self.assertEqual(rows[0]["cosine"], 0.0)
        self.assertEqual(rows[1]["cosine"], 0.0)

    def test_repeated_term_beats_jaccard(self) -> None:
        docs = ["alpha alpha alpha", "alpha beta gamma"]
        rows = rank("alpha alpha beta", docs)
        by_index = {int(row["index"]): row for row in rows}
        self.assertGreater(by_index[0]["cosine"], by_index[1]["cosine"])
        self.assertGreater(by_index[1]["jaccard"], by_index[0]["jaccard"])

    def test_one_shared_rare_term(self) -> None:
        docs = ["a a", "a b"]
        tokens = [tokenize(doc) for doc in docs]
        weights = idf_map(tokens)
        query = tfidf(tokenize("b"), weights)
        idf_b = math.log((1 + 2) / (1 + 1)) + 1.0
        self.assertAlmostEqual(weights["b"], idf_b)
        self.assertAlmostEqual(cosine(query, tfidf(tokens[0], weights)), 0.0)
        self.assertGreater(cosine(query, tfidf(tokens[1], weights)), 0.5)

    def test_jaccard_of_the_tie_case(self) -> None:
        self.assertAlmostEqual(jaccard(["alpha", "alpha", "beta"], ["alpha", "alpha", "alpha"]), 0.5)
        self.assertAlmostEqual(jaccard(["alpha", "alpha", "beta"], ["alpha", "beta", "gamma"]), 2 / 3)

    def test_empty_query(self) -> None:
        rows = rank("", ["alpha"])
        self.assertEqual(rows[0]["cosine"], 0.0)
        self.assertEqual(rows[0]["jaccard"], 0.0)


if __name__ == "__main__":
    unittest.main()
