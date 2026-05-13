import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        
        hidden = np.dot(x, w) + b

        y_out = 1 / (1 + np.exp(-hidden)) if activation == "sigmoid" else max(0.0, hidden)

        return round(y_out, 5)