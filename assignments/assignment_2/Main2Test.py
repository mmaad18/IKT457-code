import unittest

import numpy as np
from numpy.typing import NDArray

from assignments.assignment_2.Patient import Patient
from assignments.assignment_2.TsetlinClause import TsetlinClause


class MyTestCase(unittest.TestCase):
    def create_tsetlin_clause(self):
        patient = Patient("ge40", "3-5", 3, True)
        observation = patient.full_observation()
        observation_size = observation.shape[0]
        tsetlin_clause = TsetlinClause(3, observation_size)
        tsetlin_clause.states.fill(3)

        return tsetlin_clause, patient, tsetlin_clause.states


    def test_patient_1(self):
        patient = Patient("ge40", "3-5", 3, True)
        print(patient.to_dict())
        print(patient.observation())
        print(patient.full_observation())


    def test_tsetlin_clause_probability_vector(self):
        tsetlin_clause, patient, start_state = self.create_tsetlin_clause()
        prob_vector = tsetlin_clause._probability_vector(0.5)
        print(prob_vector)
        self.assertEqual(prob_vector.shape[0], len(tsetlin_clause.states))


    def test_tsetlin_clause_update(self):
        tsetlin_clause, patient, start_state = self.create_tsetlin_clause()
        rewards: NDArray[np.bool_] = np.random.rand(len(tsetlin_clause.states)) > 0.5
        tsetlin_clause.update(rewards)

        step = np.where(rewards, 1, -1)
        result_state = start_state - step  # Memory size=3 => state=3 is on the negative side => negative step
        np.testing.assert_array_equal(result_state, tsetlin_clause.states)


    def test_tsetlin_clause_memorize(self):
        tsetlin_clause, patient, start_state = self.create_tsetlin_clause()
        literals = np.random.rand(len(tsetlin_clause.states)) > 0.5
        tsetlin_clause.memorize(literals, 1.0)

        step = np.where(literals, 1, 0)
        result_state = start_state + step
        np.testing.assert_array_equal(result_state, tsetlin_clause.states)


    def test_tsetlin_clause_forget(self):
        tsetlin_clause, patient, start_state = self.create_tsetlin_clause()
        literals = np.random.rand(len(tsetlin_clause.states)) > 0.5
        tsetlin_clause.forget(literals, 1.0)

        step = np.where(literals, -1, 0)
        result_state = start_state + step
        np.testing.assert_array_equal(result_state, tsetlin_clause.states)


    def test_tsetlin_clause_type_i_feedback_condition_true(self):
        tsetlin_clause, patient, start_state = self.create_tsetlin_clause()
        observation = patient.full_observation()
        tsetlin_clause.type_i_feedback(observation, 1.0)


    def test_tsetlin_clause_type_i_feedback_condition_false(self):
        tsetlin_clause, patient, start_state = self.create_tsetlin_clause()
        observation = patient.full_observation()
        tsetlin_clause.type_i_feedback(observation, 0.0)


    def test_tsetlin_clause_type_ii_feedback(self):
        tsetlin_clause, patient, start_state = self.create_tsetlin_clause()
        observation = patient.full_observation()
        tsetlin_clause.type_ii_feedback(observation)


    def test_tsetlin_clause_vote(self):
        tsetlin_clause, patient, start_state = self.create_tsetlin_clause()
        observation = patient.full_observation()
        vote = tsetlin_clause.vote(observation)


    def test_tsetlin_clause_get_rule(self):
        tsetlin_clause, patient, start_state = self.create_tsetlin_clause()
        literals = np.random.rand(len(tsetlin_clause.states)) > 0.5
        tsetlin_clause.memorize(literals, 1.0)

        features = patient.full_features()
        rule = tsetlin_clause.get_rule(features)
        print(rule)


    def test_2(self):
        for _ in range(3):
            x = np.random.randint(-5, 5, size=2)
            print(x)


if __name__ == '__main__':
    unittest.main()
