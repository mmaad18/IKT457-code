from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class TsetlinAutomata:
    memory_size: int
    num_of_automata: int
    states: NDArray[np.int_] = field(init=False)

    def __post_init__(self) -> None:
        self.states = np.random.randint(-self.memory_size, self.memory_size + 1, size=self.num_of_automata)
        self.states[self.states == 0] = 1


    def reset(self) -> None:
        self.states = np.ones(self.num_of_automata, dtype=int)


    def update(self, rewards: NDArray[np.bool_]) -> None:
        direction = np.where(self.states > 0, 1, -1)
        step = np.where(rewards, direction, -direction)
        transition_step = np.where(self.states + step == 0, step, 0)
        self.states += step + transition_step

        self.states = np.clip(self.states, -self.memory_size, self.memory_size)


    def actions(self) -> NDArray[np.int_]:
        return (self.states > 0).astype(int)

