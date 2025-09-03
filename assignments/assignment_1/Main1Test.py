import unittest

import numpy as np

from assignments.assignment_1.TsetlinAutomata import TsetlinAutomata


class MyTestCase(unittest.TestCase):
    def test_tsetlin_automata_1(self):
        tsetlin_automata = TsetlinAutomata(3, 2)
        tsetlin_automata.states = np.array([1, 1])

        rewards = np.array([True, False])

        tsetlin_automata.update(rewards)
        expected_states = np.array([2, -1])
        np.testing.assert_array_equal(expected_states, tsetlin_automata.states)

        tsetlin_automata.update(rewards)
        expected_states = np.array([3, 1])
        np.testing.assert_array_equal(expected_states, tsetlin_automata.states)

        tsetlin_automata.update(rewards)
        expected_states = np.array([3, -1])
        np.testing.assert_array_equal(expected_states, tsetlin_automata.states)


if __name__ == '__main__':
    unittest.main()
