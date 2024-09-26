import numpy as np
import sys

class LogicGate:
    """
    A class to simulate basic logic gates using NumPy.

    Supported Gates:
    - AND
    - OR
    - NOT
    - XOR
    """

    @staticmethod
    def AND(x1, x2):
        """Returns the result of AND gate for inputs x1 and x2."""
        x1, x2 = np.array(x1), np.array(x2)
        return np.where((x1 == 1) & (x2 == 1), 1, 0)

    @staticmethod
    def OR(x1, x2):
        """Returns the result of OR gate for inputs x1 and x2."""
        x1, x2 = np.array(x1), np.array(x2)
        return np.where((x1 == 1) | (x2 == 1), 1, 0)

    @staticmethod
    def NOT(x):
        """Returns the result of NOT gate for input x."""
        x = np.array(x)
        return np.where(x == 1, 0, 1)

    @staticmethod
    def XOR(x1, x2):
        """Returns the result of XOR gate for inputs x1 and x2."""
        x1, x2 = np.array(x1), np.array(x2)
        return np.where((x1 != x2), 1, 0)

def print_help():
    help_message = """
    LogicGate Class Usage:

    - AND(x1, x2): Returns the AND operation result for two binary inputs x1 and x2.
    - OR(x1, x2): Returns the OR operation result for two binary inputs x1 and x2.
    - NOT(x): Returns the NOT operation result for a single binary input x.
    - XOR(x1, x2): Returns the XOR operation result for two binary inputs x1 and x2.

    Example Usage:

    from logic_gate import LogicGate

    # AND gate
    result = LogicGate.AND([1, 0], [1, 1])
    print(result)  # Output: [1 0]

    # OR gate
    result = LogicGate.OR([1, 0], [0, 1])
    print(result)  # Output: [1 1]

    # NOT gate
    result = LogicGate.NOT([1, 0])
    print(result)  # Output: [0 1]

    # XOR gate
    result = LogicGate.XOR([1, 0], [0, 1])
    print(result)  # Output: [1 1]
    """
    print(help_message)

if __name__ == "__main__":
    print_help()

