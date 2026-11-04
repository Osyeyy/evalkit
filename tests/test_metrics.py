import torch
import pytest 

from evalkit.metrics import cross_entropy_loss, perplexity_n

def test_cross_entropy():
    logits = torch.tensor([
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0]
    ])

    targets = torch.tensor([0, 2])

    loss = cross_entropy_loss(logits, targets)

    assert loss.item() == pytest.approx(0.407606, abs=1e-5)
    print(f"cross-entropy passed: {loss.item():.6f}")


def test_perplexity():
    loss = torch.tensor(2.0)
    ppl = perplexity_n(loss)

    assert ppl == pytest.approx(7.389056, rel=1e-5)
    print(f"Perplexity Passed: {ppl:.6f}")


if __name__ == "__main__":
    test_cross_entropy()
    test_perplexity()



