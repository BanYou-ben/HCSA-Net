"""Evaluate a local checkpoint in the customized Ultralytics environment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="outputs")
    parser.add_argument("--name", default="evaluation")
    args = parser.parse_args()
    for path in (args.weights, args.data):
        if not path.is_file():
            parser.error("File does not exist: {}".format(path))
    # Import the original customized framework that defines the checkpoint modules.
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data), split=args.split, imgsz=args.imgsz,
        batch=args.batch, device=args.device, project=args.project,
        name=args.name, plots=True,
    )
    print(json.dumps({
        "split": args.split,
        "weights": str(args.weights),
        "data": str(args.data),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }, indent=2))


if __name__ == "__main__":
    main()
