import torch

from dmme.adm import ADM


def test_adm():
    model = ADM()

    x = torch.randn(2, 3, 128, 128)
    t = torch.randint(1, 8, size=(2,))

    output = model(x, t)

    assert output.size() == x.size()
