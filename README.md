# HCSA-Net

Code resources for **HCSA-Net: Hierarchical Context Synergy and Spatially Adaptive Cross-Scale Fusion for Fine-Grained Printing Defect Detection**.

## Contents

- `configs/hcsa_net.yaml`: model architecture configuration. Historical implementation names such as `MSFE`, `Bi_FPN`, and `ASFF_Detect` are retained.
- `configs/printing.example.yaml`: example dataset configuration and printing-defect class order.
- `scripts/evaluate.py`: evaluation helper for local checkpoints in the original customized Ultralytics environment.

## Usage

Run the helper in the original customized Ultralytics environment, where the checkpoint's custom modules are registered. Prepare a local checkpoint and dataset YAML with correct image and label paths.

```bash
python scripts/evaluate.py --weights /absolute/path/to/best.pt --data /absolute/path/to/printing.yaml --split val --device cpu
```

For a separately defined test partition, use `--split test`. The helper performs evaluation only. Standard upstream Ultralytics installations do not define the custom modules referenced by the architecture configuration.

Model weights and datasets are not distributed in this repository.

## Attribution and license

The architecture configuration derives from a customized Ultralytics YOLOv8 implementation. Upstream license and attribution notices are retained. See `LICENSE` (AGPL-3.0).
