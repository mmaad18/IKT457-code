import unittest

import numpy as np

from assignments.assignment_2.Patient import Patient
from assignments.assignment_2.TsetlinMachine import TsetlinMachine


class MyTestCase(unittest.TestCase):
    def test_patient_1(self):
        patient = Patient("ge40", "3-5", 3, True)

        print(patient.to_dict())
        print(patient.to_np_array())


    def test_tsetlin_machine_condition(self):
        patient = Patient("ge40", "3-5", 3, True)
        observation_positive = patient.to_np_array()
        observation_negative = np.logical_not(observation_positive)
        observation = np.concatenate([observation_positive, observation_negative])
        observation_size = observation.shape[0]

        print(observation)

        tsetlin_machine = TsetlinMachine(3, observation_size)
        tsetlin_machine.states = np.ones(observation_size, dtype=int)

        condition = tsetlin_machine.condition(observation)
        print(condition)


    def test_2(self):
        for _ in range(100):
            x = np.random.randint(-5, 5, size=2)
            print(x)


if __name__ == '__main__':
    unittest.main()
