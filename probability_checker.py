import json
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import functools


class FakeDieDistributionBank(object):
    def __init__(self):
        pass

    def get_dist(self, index):
        return [1.]

    def get_dist_with_discard_lowest(self, index):
        return [1.]


class DistributionBank(object):

    def __init__(self, dice_size):
        self.dice_size = dice_size
        self.distributions = [[None], [0] +  [1. / dice_size] * dice_size]
        self.prepared_dists = 1

    def get_dist(self, index):
        while index > self.prepared_dists:
            self.distributions.append(np.convolve(self.distributions[self.prepared_dists], self.distributions[1]))
            self.prepared_dists += 1

        return self.distributions[index]

    def get_dist_with_discard_lowest(self, num_of_dices):
        res = [0.] * ((num_of_dices - 1) * self.dice_size + 1)

        it = [1] * num_of_dices
        keep_repeating = True
        last_loop = False

        while keep_repeating:
            res[sum(it) - min(it)] += 1
            it[0] += 1
            for i in range(len(it) - 1):
                if it[i] == (self.dice_size + 1):
                    it[i] = 1
                    it[i+1] += 1

            if last_loop:
                keep_repeating = False

            if it == ([self.dice_size] * num_of_dices):
                last_loop = True

        res = [el / (self.dice_size ** num_of_dices) for el in res]

        return res


ExpressionInfo = namedtuple('ExpressionInfo', ['dmg_name', 'hit_name'])


class ExpressionProcessor(object):

    def __init__(self, hit_probs, dmg_dists):
        self.hit_probs = hit_probs
        self.dmg_dists = dmg_dists

    def process_expression(self, exp):

        basic_arr = self._split_into_basic(exp)
        parsed_expressions = [self._parse_single_expression(el) for el in basic_arr]
        evaluated_expressions = [self._evaluate_single_expression(el) for el in parsed_expressions]

        return list(functools.reduce(self._combine_distributions, evaluated_expressions))

    def _split_into_basic(self, exp):
        raw_arr = exp.split('+')
        res = []

        for el in raw_arr:
            if '*' in el:
                splitted = el.split('*')
                num = int(splitted[0].strip())
                val = splitted[1].strip()
                res += [val] * num
            else:
                res.append(el.strip())

        return res

    def _parse_single_expression(self, exp):
        dmg_name = exp[:exp.find('[')]
        hit_name = exp[exp.find('[')+1:len(exp)-1]
        return ExpressionInfo(dmg_name=dmg_name, hit_name=hit_name)

    def _evaluate_single_expression(self, exp):
        dmg_dist = self.dmg_dists[exp.dmg_name]
        hit_prob = self.hit_probs[exp.hit_name]

        res = [el * hit_prob for el in dmg_dist[1:]]
        res = [dmg_dist[0] * hit_prob + 1 - hit_prob] + res

        return res

    def _combine_distributions(self, dist_1, dist_2):
        return np.convolve(list(dist_1), list(dist_2))


class ProbabilityChecker(object):

    def __init__(self, filename):
        self.filename = filename
        self.db = {
            'd0': FakeDieDistributionBank(),
            'd3': DistributionBank(3),
            'd6': DistributionBank(6)
        }

    def analyze(self):
        config = json.load(open(self.filename, 'r'))

        hit_probs = self._process_hit_definitions(config['hit_tests'])
        dmg_dists = self._process_dmg_distributions(config['damage_tests'])

        ep = ExpressionProcessor(hit_probs, dmg_dists)

        analysis_results = {}

        for expression in config['expressions']:
            analysis_results[expression['name']] = ep.process_expression(expression['value'])

        return analysis_results

    def present_results(self, arg):

        for exp_name, dist in arg.items():

            cumul = [0]
            for el in dist:
                cumul.append(el + cumul[-1])

            inv_cumul = [1 - el for el in cumul]

            print(exp_name, ':')

            exp_value = sum([idx * val for (idx, val) in enumerate(dist)])

            deviation = 0

            for idx, el in enumerate(dist):
                deviation += (idx - exp_value)**2 * el

            deviation = deviation**0.5

            print('expected value {:3.2f}'.format(exp_value))
            print('standard deviation {:3.2f}'.format(deviation))
            print('assumed_range <{:3.2f}, {:3.2f}>\n'.format(
                next(idx for (idx, val) in enumerate(inv_cumul) if (lambda el_idx, x: x < 0.8)(idx, val)),
                next(idx for (idx, val) in enumerate(inv_cumul) if (lambda el_idx, x: x < 0.2)(idx, val))))

            print('dmg, prob, at_least, at_most, assumed_range')

            for idx, el in enumerate(dist):
                print('{:02}, {:3.2f}, {:3.2f}, {:3.2f}'.format(idx,
                                                                100 * el,
                                                                100 * (1 - cumul[idx]),
                                                                100 * cumul[idx]))

            print()

            fig = plt.figure()
            fig.suptitle(exp_name)
            ax = fig.add_subplot(311)
            ax.set_title('Inverted cumulative distribution')
            ax.plot(inv_cumul)
            ax = fig.add_subplot(312)
            ax.set_title('Cumulative distribution')
            ax.plot(cumul)
            ax = fig.add_subplot(313)
            ax.set_title('Distribution')
            ax.plot(dist)
            fig.show()

        plt.show()

    def _process_single_hit_definition(self, hit_definition):

        if hit_definition['discard_lowest']:
            raw_dist = self.db[hit_definition['dice_size']].get_dist_with_discard_lowest(hit_definition['num_of_dices'])
        else:
            raw_dist = self.db[hit_definition['dice_size']].get_dist(hit_definition['num_of_dices'])

        l_shift_value = hit_definition['threshold'] - hit_definition['bonus']

        if l_shift_value > 0:
            res = sum(raw_dist[l_shift_value:])
        else:
            res = sum(raw_dist)

        if hit_definition['reroll']:
            res = res + (1 - res) * res

        return res

    def _process_hit_definitions(self, hit_defs):
        res = {
            'auto' : 1
        }
        for hit_definition in hit_defs:
            res[hit_definition['name']] = self._process_single_hit_definition(hit_definition)

        return res

    def _process_single_dmg_distribution(self, dmg_definition):

        if dmg_definition['discard_lowest']:
            raw_dist = self.db[dmg_definition['dice_size']].get_dist_with_discard_lowest(dmg_definition['num_of_dices'])
        else:
            raw_dist = self.db[dmg_definition['dice_size']].get_dist(dmg_definition['num_of_dices'])

        l_shift_value = dmg_definition['threshold'] - dmg_definition['bonus']

        if l_shift_value > 0:
            res = [sum(raw_dist[:l_shift_value+1])] + list(raw_dist[l_shift_value+1:])
        else:
            res = [0] * (-l_shift_value) + list(raw_dist)

        return res

    def _process_dmg_distributions(self, dmg_defs):
        res = {}
        for dmg_definition in dmg_defs:
            res[dmg_definition['name']] = self._process_single_dmg_distribution(dmg_definition)

        return res


if __name__ == '__main__':

    pc = ProbabilityChecker('config.json')
    results = pc.analyze()
    pc.present_results(results)