import unittest

import numpy as np
from numpy.typing import NDArray

from assignments.assignment_2.Patient import Patient
from assignments.assignment_2.TsetlinMachine import TsetlinMachine


class MyTestCase(unittest.TestCase):
    def create_tsetlin_machine(self):
        patient = Patient("ge40", "3-5", 3, True)
        observation = patient.full_observation()
        observation_size = observation.shape[0]
        tsetlin_machine = TsetlinMachine(3, observation_size)
        tsetlin_machine.states = np.ones(observation_size, dtype=int) + 2

        return tsetlin_machine, patient, tsetlin_machine.states


    def test_patient_1(self):
        patient = Patient("ge40", "3-5", 3, True)
        print(patient.to_dict())
        print(patient.observation())
        print(patient.full_observation())


    def test_tsetlin_machine_probability_vector(self):
        tsetlin_machine, patient, start_state = self.create_tsetlin_machine()
        prob_vector = tsetlin_machine._probability_vector(0.5)
        print(prob_vector)
        self.assertEqual(prob_vector.shape[0], len(tsetlin_machine.states))


    def test_tsetlin_machine_update(self):
        tsetlin_machine, patient, start_state = self.create_tsetlin_machine()
        rewards: NDArray[np.bool_] = np.random.rand(len(tsetlin_machine.states)) > 0.5
        tsetlin_machine.update(rewards)

        step = np.where(rewards, 1, -1)
        result_state = start_state - step  # Memory size=3 => state=3 is on the negative side => negative step
        np.testing.assert_array_equal(result_state, tsetlin_machine.states)


    def test_tsetlin_machine_memorize(self):
        tsetlin_machine, patient, start_state = self.create_tsetlin_machine()
        literals = np.random.rand(len(tsetlin_machine.states)) > 0.5
        tsetlin_machine.memorize(literals, 1.0)

        step = np.where(literals, 1, 0)
        result_state = start_state + step
        np.testing.assert_array_equal(result_state, tsetlin_machine.states)


    def test_tsetlin_machine_forget(self):
        tsetlin_machine, patient, start_state = self.create_tsetlin_machine()
        literals = np.random.rand(len(tsetlin_machine.states)) > 0.5
        tsetlin_machine.forget(literals, 1.0)

        step = np.where(~literals, -1, 0)
        result_state = start_state + step
        np.testing.assert_array_equal(result_state, tsetlin_machine.states)


    def test_tsetlin_machine_type_i_feedback(self):
        tsetlin_machine, patient, start_state = self.create_tsetlin_machine()
        observation = patient.full_observation()
        tsetlin_machine.type_i_feedback(observation, 0.5)


    def test_2(self):
        for _ in range(100):
            x = np.random.randint(-5, 5, size=2)
            print(x)


if __name__ == '__main__':
    unittest.main()
