import unittest

import numpy as np

from assignments.assignment_1.TsetlinAutomata import TsetlinAutomata


class MyTestCase(unittest.TestCase):
    def test_tsetlin_automata_1(self):
        tsetlin_automata = TsetlinAutomata(3, 2)
        tsetlin_automata.states = np.array([4, 4])

        rewards = np.array([True, False])

        tsetlin_automata.update(rewards)
        expected_states = np.array([5, 3])
        np.testing.assert_array_equal(expected_states, tsetlin_automata.states)

        tsetlin_automata.update(rewards)
        expected_states = np.array([6, 4])
        np.testing.assert_array_equal(expected_states, tsetlin_automata.states)

        tsetlin_automata.update(rewards)
        expected_states = np.array([6, 3])
        np.testing.assert_array_equal(expected_states, tsetlin_automata.states)


    def test_2(self):
        for _ in range(100):
            x = np.random.randint(-5, 5, size=2)
            print(x)


if __name__ == '__main__':
    unittest.main()
