class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:

        curr_error = init

        for _ in range(iterations):
            derivative = 2 * curr_error
            curr_error = curr_error - learning_rate * derivative

        return round(curr_error, 5)
