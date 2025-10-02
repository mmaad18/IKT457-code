from random import choice

import numpy as np

from assignments.assignment_2.Patient import Patient
from assignments.assignment_2.TsetlinClause import TsetlinClause


def patient_data() -> list[Patient]:
    return [
        Patient("ge40", "3-5", 3, True),
        Patient("lt40", "0-2", 3, False),
        Patient("ge40", "6-8", 3, True),
        Patient("ge40", "0-2", 2, False),
        Patient("premeno", "0-2", 3, True),
        Patient("premeno", "0-2", 1, False)
    ]


def R1(p: Patient) -> bool:
    return p.Deg_malig == 3 and p.Menopause != "lt40"


def R2(p: Patient) -> bool:
    return p.Deg_malig == 3


def R3(p: Patient) -> bool:
    return p.Inv_nodes == "0-2"


def R_classification(p: Patient) -> tuple[bool, bool, bool]:
    return R1(p), R2(p), R3(p)


def learn_rule(
    memorize_value: float,
    learn_recurrence: bool,
    memory_size: int = 10,
    epochs: int = 50,
    steps_per_epoch: int = 10,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)

    patients = [
        Patient("ge40",    "3-5", 3, True),
        Patient("lt40",    "0-2", 3, False),
        Patient("ge40",    "6-8", 3, True),
        Patient("ge40",    "0-2", 2, False),
        Patient("premeno", "0-2", 3, True),
        Patient("premeno", "0-2", 1, False),
    ]

    full_features = patients[0].full_features()
    observations  = [p.full_observation() for p in patients]

    idx_pos = [i for i, p in enumerate(patients) if p.Recurrence]
    idx_neg = [i for i, p in enumerate(patients) if not p.Recurrence]

    clause = TsetlinClause(memory_size=memory_size, num_of_automata=len(full_features))
    clause.states.fill(memory_size // 2)

    # Training loop (balanced: each step uses one positive + one negative)
    for _ in range(epochs):
        # Shuffle locally each epoch for balance
        rng.shuffle(idx_pos)
        rng.shuffle(idx_neg)

        for k in range(steps_per_epoch):
            i_pos = idx_pos[k % len(idx_pos)]
            i_neg = idx_neg[k % len(idx_neg)]
            x_pos = observations[i_pos]
            x_neg = observations[i_neg]

            if learn_recurrence:
                clause.type_i_feedback(x_pos, memorize_value)
                clause.type_ii_feedback(x_neg)
            else:
                clause.type_i_feedback(x_neg, memorize_value)
                clause.type_ii_feedback(x_pos)

    rule_type = "Recurrence" if learn_recurrence else "Non-recurrence"
    rule_str  = f"IF {clause.get_rule(full_features)} THEN {rule_type}"
    print(rule_str)


def main():
    patients = patient_data()
    R_classifications = [R_classification(p) for p in patients]
    print(R_classifications)

    learn_rule(0.2, True)
    #learn_rule(0.2, learn_for_recurrence=False)


if __name__ == '__main__':
    main()
