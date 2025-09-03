import numpy as np
from numpy.typing import NDArray

from assignments.assignment_1.TsetlinAutomata import TsetlinAutomata


def environment_1(actions: NDArray[np.int_]) -> NDArray[np.bool_]:
    M = actions.sum()

    if M < 4:
        reward_probability = M * 0.2
        return np.random.choice([True, False], size=len(actions), p=[reward_probability, 1 - reward_probability])
    else:
        reward_probability = 0.6 - (M - 3) * 0.2
        return np.random.choice([True, False], size=len(actions), p=[reward_probability, 1 - reward_probability])


def main():
    tsetlin_automata = TsetlinAutomata(3, 5)

    for _ in range(20):
        actions = tsetlin_automata.actions()
        rewards = environment_1(actions)
        states = tsetlin_automata.states
        tsetlin_automata.update(rewards)
        print(f"States: {states}, Actions: {actions}, Rewards: {rewards}")


if __name__ == '__main__':
    main()

