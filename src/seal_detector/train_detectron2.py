"""Train a Mask R-CNN seal segmenter with modern Detectron2.

This entrypoint is intended for a current Detectron2 install built against the
active PyTorch environment, rather than the older Detectron2 0.5 release line.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-json", type=Path)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument(
        "--config-name",
        default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        help="Detectron2 Model Zoo config name.",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--base-lr", type=float, default=0.00025)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultTrainer
    except ImportError as exc:
        raise SystemExit(
            "detectron2 is required. Install a current source build compatible with your active PyTorch stack."
        ) from exc

    train_name = "seal_train"
    val_name = "seal_val"

    register_coco_instances(train_name, {}, str(args.train_json), str(args.train_images))

    datasets_test: tuple[str, ...] = ()
    if args.val_json and args.val_images:
        register_coco_instances(val_name, {}, str(args.val_json), str(args.val_images))
        datasets_test = (val_name,)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config_name))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = datasets_test
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config_name)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.MODEL.DEVICE = args.device
    cfg.INPUT.MASK_FORMAT = "polygon"
    cfg.OUTPUT_DIR = str(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.yaml").write_text(cfg.dump())
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
