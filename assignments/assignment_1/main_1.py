import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from assignments.assignment_1.TsetlinAutomata import TsetlinAutomata


def plot_M(M_list: list[int], memory_size: int, num_of_automata: int) -> None:
    plt.figure()
    plt.plot(M_list)

    plt.title(f"M over time (N={num_of_automata}, n={memory_size})")
    plt.xlabel("Step")
    plt.ylabel("M")

    plt.show()


def environment_1(actions: NDArray[np.int_]) -> NDArray[np.bool_]:
    M = actions.sum()

    if M < 4:
        reward_probability = M * 0.2
        return np.random.choice([True, False], size=len(actions), p=[reward_probability, 1 - reward_probability])
    else:
        reward_probability = 0.6 - (M - 3) * 0.2
        return np.random.choice([True, False], size=len(actions), p=[reward_probability, 1 - reward_probability])


def main():
    memory_size = 50
    num_of_automata = 5
    tsetlin_automata = TsetlinAutomata(memory_size, num_of_automata)

    M_list = []

    for _ in range(1000):
        actions = tsetlin_automata.get_actions()
        rewards = environment_1(actions)
        states = tsetlin_automata.get_states()
        tsetlin_automata.update(rewards)
        print(f"States: {states}, Actions: {actions}, Rewards: {rewards}")
        M_list.append(actions.sum())

    plot_M(M_list, memory_size, num_of_automata)


if __name__ == '__main__':
    main()

