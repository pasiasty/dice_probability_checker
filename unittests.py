import unittest
from probability_checker import *
import functools


def compare_lists(test_case, list_a, list_b):
    test_case.assertEqual(len(list_a), len(list_b))
    test_case.assertAlmostEqual(list_a, list_b)


class DistributionBankTests(unittest.TestCase):

    def setUp(self):
        self.db = DistributionBank(6)

    def test_get_1d6_dist(self):
        exp_res = [0.] + [1./6] * 6
        compare_lists(self, exp_res, self.db.get_dist(1))

    def test_dist_with_discard_lowest_2d6(self):
        exp_res = [0.0, 0.027777777777777776, 0.08333333333333333,
                   0.1388888888888889, 0.19444444444444445, 0.25, 0.3055555555555556]
        compare_lists(self, exp_res, self.db.get_dist_with_discard_lowest(2))

    def test_dist_with_discard_lowest_3d6(self):
        exp_res = [0.0, 0.0, 0.004629629629629629, 0.013888888888888888, 0.032407407407407406, 0.05555555555555555,
                   0.08796296296296297, 0.125, 0.1574074074074074, 0.16666666666666666,
                   0.1574074074074074, 0.125, 0.07407407407407407]

        compare_lists(self, exp_res, self.db.get_dist_with_discard_lowest(3))


class ExpressionProcessorTests(unittest.TestCase):

    def setUp(self):
        hit_probs = {
            'auto': 1,
            'simple': 0.8
        }
        dmg_dists = {
            '1_dmg': [0., 1.],
            'd6_dmg': [0.] + [1. / 6] * 6
        }
        self.ep = ExpressionProcessor(hit_probs, dmg_dists)

    def test_split_into_basic(self):
        arg = 'dmg_1[auto] + 2*dmg_1[hit_1]'
        exp_res = ['dmg_1[auto]', 'dmg_1[hit_1]', 'dmg_1[hit_1]']

        self.assertEqual(exp_res, self.ep._split_into_basic(arg))

    def test_parsing_single_expression(self):
        arg = 'dmg_1[auto]'
        exp_res = ExpressionInfo(dmg_name='dmg_1', hit_name='auto')

        self.assertEqual(exp_res, self.ep._parse_single_expression(arg))

    def test_evaluate_single_expression_auto_hit(self):
        arg = ExpressionInfo(dmg_name='1_dmg', hit_name='auto')
        exp_res = [0.0, 1.0]
        self.assertEqual(exp_res, self.ep._evaluate_single_expression(arg))

    def test_evaluate_single_expression_simple_hit(self):
        arg = ExpressionInfo(dmg_name='d6_dmg', hit_name='simple')
        exp_res = [0.19999999999999996, 0.13333333333333333, 0.13333333333333333, 0.13333333333333333,
                   0.13333333333333333, 0.13333333333333333, 0.13333333333333333]
        self.assertEqual(exp_res, self.ep._evaluate_single_expression(arg))

    def test_process_expression(self):
        arg = '1_dmg[auto] + 2*d6_dmg[simple]'
        exp_res = [0., 0.04, 0.05333333,  0.07111111,  0.08888889,  0.10666667,
                   0.12444444, 0.14222222, 0.10666667, 0.08888889, 0.07111111, 0.05333333, 0.03555556, 0.01777778]
        res = list(self.ep.process_expression(arg))

        def reduction_function(accum, vals):
            if not accum:
                return None

            accum.assertAlmostEqual(vals[0], vals[1])
            return accum

        functools.reduce(reduction_function, zip(res, exp_res), self)


class ProcessHitDefinitionsTests(unittest.TestCase):

    def setUp(self):
        self.pc = ProbabilityChecker('')

    def test_1d6_thr_4(self):
        arg = {
            "num_of_dices": 1,
            "dice_size": 'd6',
            "bonus": 0,
            "threshold": 4,
            "discard_lowest": False,
            "reroll": False
        }

        self.assertAlmostEqual(0.5, self.pc._process_single_hit_definition(arg))

    def test_1d6_thr_4_reroll(self):
        arg = {
            "num_of_dices": 1,
            "dice_size": 'd6',
            "bonus": 0,
            "threshold": 4,
            "discard_lowest": False,
            "reroll": True
        }

        self.assertAlmostEqual(0.75, self.pc._process_single_hit_definition(arg))

    def test_3d6_thr_4_discard_lowest(self):
        arg = {
            "num_of_dices": 3,
            "dice_size": 'd6',
            "bonus": 6,
            "threshold": 14,
            "discard_lowest": True,
            "reroll": False
        }

        self.assertAlmostEqual(0.6805555555555556, self.pc._process_single_hit_definition(arg))

    def test_1d6_dmg(self):
        arg = {
            "name": "dmg_1",
            "num_of_dices": 1,
            "dice_size": 'd6',
            "bonus": 2,
            "threshold": 5,
            "discard_lowest": False
        }

        exp_res = [0.5, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666]

        res = self.pc._process_single_dmg_distribution(arg)
        compare_lists(self, exp_res, res)

    def test_1d0_dmg(self):
        arg = {
            "name": "dmg_1",
            "num_of_dices": 1,
            "dice_size": 'd0',
            "bonus": 5,
            "threshold": 3,
            "discard_lowest": False
        }

        exp_res = [0, 0, 1]

        res = self.pc._process_single_dmg_distribution(arg)
        compare_lists(self, exp_res, res)

if __name__ == '__main__':
    unittest.main()
