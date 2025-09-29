from assignments.assignment_2.Patient import Patient


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


def main():
    patients = patient_data()
    R_classifications = [R_classification(p) for p in patients]
    print(R_classifications)


if __name__ == '__main__':
    main()
