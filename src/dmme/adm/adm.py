from torch import nn


class ADM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, c):
        return x


class ADMG(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, c):
        return x


class ADMU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, c):
        return x
