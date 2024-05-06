from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class LearningRate:
    """
    A wrapper class for a learning rate. For unknown reasons, using python
    floats to represent a learning rate results in small errors in the learning rate
    value within the optimizer across instantiations of the same experiment. For example, lr may be 1e-5
    but when instantiating the experiment again it may be 9.99999999e-6. This results in the hash
    for the same experiment to differ across instantiations. This wrapper class is created to work
    with the nice_logspace function and avoid the hashing issue stated above.
    """

    exponent: Fraction
    base: int = 10

    def as_float(self) -> float:
        """The value of the learning rate to be used within the optimizer

        Returns:
            float: the float value of the learning rate
        """
        return float(self.base**self.exponent)

    def __str__(self) -> str:
        """Used for the hash

        Returns:
            str: The string representation of the learning rate
        """
        return f"{self.base}^{self.exponent}"

    def as_latex_str(self) -> str:
        """Used for the plot labels

        Returns:
            str: The string representation of the learning rate
        """
        return f"${self.base}^{{{self.exponent}}}$"

    def __lt__(self, other):
        return self.as_float() < other.as_float()

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return self.as_float() > other.as_float()

    def __ge__(self, other):
        return self == other or self > other
