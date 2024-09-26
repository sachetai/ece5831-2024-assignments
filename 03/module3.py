from logic_gate import LogicGate

# AND gate example
print(LogicGate.AND([1, 0], [1, 1]))  # Output: [1 0]

# OR gate example
print(LogicGate.OR([1, 0], [0, 1]))   # Output: [1 1]

# NOT gate example
print(LogicGate.NOT([1, 0]))          # Output: [0 1]

# XOR gate example
print(LogicGate.XOR([1, 0], [0, 1]))  # Output: [1 1]
