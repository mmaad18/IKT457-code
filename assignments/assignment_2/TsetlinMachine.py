from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class TsetlinMachine:
    memory_size: int
    num_of_automata: int
    states: NDArray[np.int_] = field(init=False)

    def __post_init__(self) -> None:
        self.states = np.random.randint(1, 2*self.memory_size + 1, size=self.num_of_automata)


    def reset(self) -> None:
        self.states = np.ones(self.num_of_automata, dtype=int)


    def update(self, rewards: NDArray[np.bool_]) -> None:
        direction = np.where(self.states > self.memory_size, 1, -1)
        step = np.where(rewards, direction, -direction)
        self.states += step

        self.states = np.clip(self.states, 1, 2*self.memory_size)


    def get_actions(self) -> NDArray[np.bool_]:
        return self.states > self.memory_size


    def get_states(self) -> NDArray[np.int_]:
        return self.states.copy()


    def condition(self, observation: NDArray[np.bool_]) -> np.bool_:
        return np.all(observation[self.states > self.memory_size])


    def type_i_feedback(self, observation: NDArray[np.bool_], memorize_value: float) -> None:
        forget_value = 1 - memorize_value
        condition = self.condition(observation)
        if condition:
            self.states = np.where(observation, forget_value, self.memory_size)
        else:
            self.states = np.where(observation, forget_value, self.memory_size + 1)


    def erase_feedback(self, memory: NDArray[np.int_]) -> None:
        self.states = np.where(memory == 1, self.memory_size + 1, self.memory_size)


    def type_ii_feedback(self) -> None:
        self.states = np.where(self.states > self.memory_size, self.memory_size, self.memory_size + 1)

