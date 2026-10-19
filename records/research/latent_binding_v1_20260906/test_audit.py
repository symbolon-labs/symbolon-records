"""Adversarial tests for the independent reporting audit (no model execution)."""
import copy
import unittest

import audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [a + str(d) for a in 'ABCDEFGHIJKLMNOP' for d in range(8)]
        parts = [[-float(i + 1), -0.25] for i in range(128)]
        self.record = dict(token_log_probabilities=parts,
                           answer_log_probabilities=[sum(p) for p in parts],
                           prediction='A0')

    def test_valid_prediction(self):
        self.assertEqual(audit.prediction(self.record, self.candidates), 'A0')

    def test_prediction_metadata_cannot_override_scores(self):
        record = copy.deepcopy(self.record)
        record['prediction'] = 'P7'
        with self.assertRaises(AssertionError):
            audit.prediction(record, self.candidates)

    def test_missing_second_token_not_accepted(self):
        record = copy.deepcopy(self.record)
        record['answer_log_probabilities'][0] = record['token_log_probabilities'][0][0]
        with self.assertRaises(AssertionError):
            audit.prediction(record, self.candidates)

    def test_nonfinite_score_not_accepted(self):
        record = copy.deepcopy(self.record)
        record['answer_log_probabilities'][10] = float('nan')
        with self.assertRaises(AssertionError):
            audit.prediction(record, self.candidates)

    def test_source_order_not_answer_order(self):
        source = 'D3 = yellow\nA0 = red\nC2 = blue\nB1 = green'
        self.assertEqual(audit.source_answers(source)['green'], 'B1')
        with self.assertRaises(AssertionError):
            audit.source_answers(source.replace('B1 = green', 'A0 = green'))


if __name__ == '__main__':
    unittest.main()
