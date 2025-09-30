from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Patient:
    Menopause: str
    Inv_nodes: str
    Deg_malig: int
    Recurrence: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "Menopause_lt40": self.Menopause == "lt40",
            "Menopause_ge40": self.Menopause == "ge40",
            "Menopause_premeno": self.Menopause == "premeno",
            "Inv_nodes_0_2": self.Inv_nodes == "0-2",
            "Inv_nodes_3_5": self.Inv_nodes == "3-5",
            "Inv_nodes_6_8": self.Inv_nodes == "6-8",
            "Deg_malig_1": self.Deg_malig == "1",
            "Deg_malig_2": self.Deg_malig == "2",
            "Deg_malig_3": self.Deg_malig == "3",
            "Recurrence": self.Recurrence,
        }


    def observation(self) -> NDArray[np.bool_]:
        return np.array([
            self.Menopause == "lt40",
            self.Menopause == "ge40",
            self.Menopause == "premeno",
            self.Inv_nodes == "0-2",
            self.Inv_nodes == "3-5",
            self.Inv_nodes == "6-8",
            self.Deg_malig == "1",
            self.Deg_malig == "2",
            self.Deg_malig == "3",
            self.Recurrence,
        ])


    def full_observation(self) -> NDArray[np.bool_]:
        return np.concatenate([self.observation(), ~self.observation()])

