"""
"""
import pytest

import torch


@pytest.mark.parametrize(
    "inp,tgt,b_i,acc",
    [
        (torch.tensor([[0,0,0,0],[0,0,0,0]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), -1, 0.375),
        (torch.tensor([[0,0,0,0],[0,0,0,0]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), 0, 0.375),
        (torch.tensor([[0,0,0,0],[0,0,0,0]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), 1, 0.375),
    ])
def test_measure_accuracy(inp, tgt, b_i, acc):
    assert measure_accuracy(inp, tgt, b_i) == pytest.approx(acc)


@pytest.mark.parametrize(
    "inp,tgt,b_i,prec",
    [
        (torch.tensor([[0,0,0,1],[0,0,0,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), -1, 0.5),
        (torch.tensor([[0,0,0,1],[0,0,0,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), 0, 0.5),
        (torch.tensor([[0,0,0,1],[0,0,0,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), 1, .125),
        (torch.tensor([[0,0,1,1],[0,0,1,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), -1, .75),
        (torch.tensor([[0,0,1,1],[0,0,1,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), 0, .75),
        (torch.tensor([[0,0,1,1],[0,0,1,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), 1, .375),
    ])
def test_measure_precision(inp, tgt, b_i, prec):
    assert measure_precision(inp, tgt, b_i) == pytest.approx(prec)

@pytest.mark.parametrize(
    "inp,tgt,b_i,prec",
    [
        (torch.tensor([[0,0,0,1],[0,0,0,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), -1, 0.2),
        (torch.tensor([[0,0,0,1],[0,0,0,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), 0, 1/6),
        (torch.tensor([[0,0,0,1],[0,0,0,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), 1, .25),
        (torch.tensor([[0,0,1,1],[0,0,1,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), -1, 3/5),
        (torch.tensor([[0,0,1,1],[0,0,1,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), 0, (1/2+2/3)/2),
        (torch.tensor([[0,0,1,1],[0,0,1,1]]), torch.tensor([[0,1,1,0],[0,1,1,1]]), 1, (0+0+1+1)/4),
    ])
def test_measure_recall(inp, tgt, b_i, prec):
    assert measure_recall(inp, tgt, b_i) == pytest.approx(prec)