import torch

from dmme.adm import ADM, ADMG, ADMU


def test_adm():
    model = ADM()

    x = torch.randn(2, 3, 128, 128)
    t = torch.randint(1, 8, size=(2,))

    output = model(x, t)

    assert output.size() == x.size()


def test_adm_g():
    model = ADMG()

    x = torch.randn(2, 3, 128, 128)
    t = torch.randint(1, 8, size=(2,))

    output = model(x, t)

    assert output.size() == x.size()


def test_adm_u():
    model = ADMU()

    x = torch.randn(2, 3, 128, 128)
    t = torch.randint(1, 8, size=(2,))

    output = model(x, t)

    assert output.size() == x.size()
