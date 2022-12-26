def parse(pattern: str):
    left, right = pattern.split("->")

    inputs = left.split(" ")
    output = right.strip()

    return inputs, output
