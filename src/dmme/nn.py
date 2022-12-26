from torch import nn


class Add(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


class Subtract(nn.Module):
    def forward(self, a, b):
        return a - b


class Multiply(nn.Module):
    def forward(self, a, b):
        return a * b


class Divide(nn.Module):
    def forward(self, a, b):
        return a / b


def at(dictionary, keys):
    return [dictionary.get(key) for key in keys]


class DictModule(nn.Module):
    def __init__(self, module, input_keys, output_key) -> None:
        super().__init__()

        self.module = module

        self.input_keys = input_keys
        self.output_key = output_key

    def forward(self, **kwargs):
        if isinstance(self.module, Sequential):
            kwargs[self.output_key] = self.module(**kwargs)
        else:
            input_args = at(kwargs, self.input_keys)
            kwargs[self.output_key] = self.module(*input_args)
        return kwargs


def parse(pattern: str):
    left, right = pattern.split("->")

    inputs = left.split(" ")[:-1]
    output = right.strip()
    assert output != ""

    return inputs, output


class Sequential(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()

        modules = []

        for module, pattern in args:
            inputs, output = parse(pattern)
            modules.append(DictModule(module, inputs, output))

        self.output = output
        self.layers = nn.ModuleList(modules)

    def forward(self, **kwargs):
        for f in self.layers:
            kwargs = f(**kwargs)

        return kwargs[self.output]
