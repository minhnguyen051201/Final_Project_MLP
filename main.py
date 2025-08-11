import random
import torch
from easydict import EasyDict
from utils.dirs import create_dirs
from utils.config import process_config
from utils.utils import get_args, get_logger

from models.aux_model import AuxModel
from data.data_loader import get_train_val_dataloader
from data.data_loader import get_target_dataloader
from data.data_loader import get_test_dataloader


def main():
    args = get_args()
    config = process_config(args.config)

    # set mps for macos
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    config.device = str(device)
    print(f"[INFO] Using device: {config.device}")

    # create the experiments dirs
    create_dirs([config.cache_dir, config.model_dir, config.log_dir, config.img_dir])

    # logging to the file and stdout
    logger = get_logger(config.log_dir, config.exp_name)

    # Ensure a method is defined; infer a sensible default if missing
    if not hasattr(config, "method"):
        inferred = None
        try:
            if getattr(getattr(config, "PRETEXT", EasyDict()), "USE_ROTATION", False):
                inferred = "rotate"
        except Exception:
            pass
        if not inferred:
            inferred = "src"
        config.method = inferred
        logger.info(f"Inferred method: {config.method}")

    # fix random seed to reproduce results (default to 42 if not provided)
    seed = getattr(config, "random_seed", 42)
    random.seed(seed)
    try:
        logger.info("Random seed: {:d}".format(int(seed)))
    except Exception:
        logger.info(f"Random seed: {seed}")

    if config.method in ["src", "jigsaw", "rotate"]:
        model = AuxModel(config, logger)
    else:
        raise ValueError("Unknown method: %s" % config.method)

    # Build datasets block if missing, using DATA/SOLVER/PRETEXT
    if not hasattr(config, "datasets"):
        try:
            data = getattr(config, "DATA", EasyDict())
            solver = getattr(config, "SOLVER", EasyDict())
            pretext = getattr(config, "PRETEXT", EasyDict())
            task_type = (
                "rotate" if getattr(pretext, "USE_ROTATION", False) else "jigsaw"
            )
            batch_size = int(getattr(solver, "BATCH_SIZE", 32))
            aux_classes = None
            angles = getattr(pretext, "ANGLES", None)
            if angles and isinstance(angles, (list, tuple)) and len(angles) >= 2:
                aux_classes = max(len(angles) - 1, 1)
            if aux_classes is None:
                aux_classes = 3
            normalize = EasyDict(
                {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            )
            img_transform = EasyDict(
                {
                    "jitter": 0,
                    "random_horiz_flip": 0,
                    "random_resize_crop": EasyDict(
                        {"size": [224, 224], "scale": [0.8, 1.0]}
                    ),
                }
            )
            src_name = [str(getattr(data, "SOURCE", "photo"))]
            tar_name = str(getattr(data, "TARGET", "sketch"))
            config.datasets = EasyDict(
                {
                    "src": EasyDict(
                        {
                            "type": task_type,
                            "name": src_name,
                            "root": str(getattr(data, "PATH", "datasets")),
                            "batch_size": batch_size,
                            "num_workers": 4,
                            "limit": False,
                            "val_size": 0.1,
                            "bias_whole_image": 0.7,
                            "aux_classes": aux_classes,
                            "img_transform": img_transform,
                            "normalize": normalize,
                        }
                    ),
                    "tar": EasyDict(
                        {
                            "type": task_type,
                            "name": tar_name,
                            "root": str(getattr(data, "PATH", "datasets")),
                            "batch_size": batch_size,
                            "num_workers": 4,
                            "limit": False,
                            "bias_whole_image": 0.7,
                            "aux_classes": aux_classes,
                            "img_transform": img_transform,
                            "normalize": normalize,
                        }
                    ),
                    "test": EasyDict(
                        {
                            "type": task_type,
                            "name": tar_name,
                            "root": str(getattr(data, "PATH", "datasets")),
                            "batch_size": batch_size,
                            "num_workers": 4,
                            "limit": False,
                            "aux_classes": aux_classes,
                            "img_transform": EasyDict(
                                {
                                    "jitter": 0,
                                    "random_resize_crop": EasyDict(
                                        {"size": [224, 224]}
                                    ),
                                }
                            ),
                            "normalize": normalize,
                        }
                    ),
                }
            )
            logger.info("Constructed datasets configuration from DATA/SOLVER/PRETEXT")
        except Exception:
            # Minimal fallback to avoid attribute errors
            config.datasets = EasyDict(
                {
                    "src": EasyDict(
                        {
                            "type": "rotate",
                            "name": ["photo"],
                            "batch_size": 32,
                            "num_workers": 4,
                            "limit": False,
                            "val_size": 0.1,
                            "bias_whole_image": 0.7,
                            "aux_classes": 3,
                            "img_transform": EasyDict(
                                {
                                    "jitter": 0,
                                    "random_horiz_flip": 0,
                                    "random_resize_crop": EasyDict(
                                        {"size": [224, 224], "scale": [0.8, 1.0]}
                                    ),
                                }
                            ),
                            "normalize": EasyDict(
                                {
                                    "mean": [0.485, 0.456, 0.406],
                                    "std": [0.229, 0.224, 0.225],
                                }
                            ),
                        }
                    ),
                    "tar": EasyDict(
                        {
                            "type": "rotate",
                            "name": "sketch",
                            "batch_size": 32,
                            "num_workers": 4,
                            "limit": False,
                            "bias_whole_image": 0.7,
                            "aux_classes": 3,
                            "img_transform": EasyDict(
                                {
                                    "jitter": 0,
                                    "random_horiz_flip": 0,
                                    "random_resize_crop": EasyDict(
                                        {"size": [224, 224], "scale": [0.8, 1.0]}
                                    ),
                                }
                            ),
                            "normalize": EasyDict(
                                {
                                    "mean": [0.485, 0.456, 0.406],
                                    "std": [0.229, 0.224, 0.225],
                                }
                            ),
                        }
                    ),
                    "test": EasyDict(
                        {
                            "type": "rotate",
                            "name": "sketch",
                            "batch_size": 32,
                            "num_workers": 4,
                            "limit": False,
                            "aux_classes": 3,
                            "img_transform": EasyDict(
                                {
                                    "jitter": 0,
                                    "random_resize_crop": EasyDict(
                                        {"size": [224, 224]}
                                    ),
                                }
                            ),
                            "normalize": EasyDict(
                                {
                                    "mean": [0.485, 0.456, 0.406],
                                    "std": [0.229, 0.224, 0.225],
                                }
                            ),
                        }
                    ),
                }
            )

    src_loader, val_loader = get_train_val_dataloader(config.datasets.src)
    test_loader = get_test_dataloader(config.datasets.test)

    tar_loader = None
    if config.datasets.get("tar", None):
        tar_loader = get_target_dataloader(config.datasets.tar)

    if config.mode == "train":
        model.train(src_loader, tar_loader, val_loader, test_loader)
    elif config.mode == "test":
        model.test(test_loader)


if __name__ == "__main__":
    main()
