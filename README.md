# HCSA-Net

Code resources for **HCSA-Net: Hierarchical Context Synergy and Spatially Adaptive Cross-Scale Fusion for Fine-Grained Printing Defect Detection**.

## Code availability

The complete source code will be made publicly available in this repository upon formal acceptance of the paper. During the review process, this repository provides a feature-fusion unit, model configurations, and an evaluation helper.

## Contents

- `modules/BiFPN.py`: the original `Bi_FPN` feature-fusion unit and its `swish` dependency.
- `demo_bifpn.py`: a standalone example using synthetic tensors.

- `configs/hcsa_net.yaml`: model architecture configuration. Historical implementation names such as `MSFE`, `Bi_FPN`, and `ASFF_Detect` are retained.
- `configs/printing.example.yaml`: example dataset configuration and printing-defect class order.
- `scripts/evaluate.py`: evaluation helper for local checkpoints in the original customized Ultralytics environment.

## Feature-fusion example

With PyTorch installed, run from the repository root:

```bash
python demo_bifpn.py
```

The example fuses two and three input tensors of shape `(1, 16, 32, 32)` and prints the output shapes. It does not load checkpoints, access data, or train a model.

`Bi_FPN(length=n)` expects `n` tensors with the same shape. The source preserves the original project formula: each learned weight is divided by the sum of the swish-transformed weights plus epsilon. This is the feature-fusion unit; the surrounding cross-scale network wiring is described in the architecture configuration.

## Evaluation

Run the helper in the original customized Ultralytics environment, where the checkpoint's custom modules are registered. Prepare a local checkpoint and dataset YAML with correct image and label paths.

```bash
python scripts/evaluate.py --weights /absolute/path/to/best.pt --data /absolute/path/to/printing.yaml --split val --device cpu
```

For a separately defined test partition, use `--split test`. The helper performs evaluation only. Standard upstream Ultralytics installations do not define the custom modules referenced by the architecture configuration.

Model weights and datasets are not distributed in this repository.

## Attribution and license

The architecture configuration derives from a customized Ultralytics YOLOv8 implementation. Upstream license and attribution notices are retained. See `LICENSE` (AGPL-3.0).
