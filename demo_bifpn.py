"""Run the feature-fusion unit on synthetic tensors; no dataset or checkpoint needed."""
import torch

from modules import Bi_FPN


def main():
    torch.manual_seed(0)
    for branches in (2, 3):
        features = [torch.randn(1, 16, 32, 32) for _ in range(branches)]
        fusion = Bi_FPN(length=branches)
        with torch.no_grad():
            output = fusion(features)
        print("{} inputs: {} -> {}".format(
            branches, [tuple(x.shape) for x in features], tuple(output.shape)
        ))


if __name__ == "__main__":
    main()
