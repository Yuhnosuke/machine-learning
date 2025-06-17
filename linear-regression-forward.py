import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_model_prediction(
        self, X: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        prediction = np.matmul(X, weights)
        return np.round(prediction, 5)

    def get_error(
        self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]
    ) -> float:
        diff = model_prediction - ground_truth
        mean_squared_error = np.mean(np.square(diff))
        return round(mean_squared_error, 5)
