import logging
import os

import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, mean_squared_error

from architectures import get_omniglot_model, get_sinusoid_model
from datasets.omniglot import OmniglotNShot
from datasets.sinusoid import SinusoidNShot


def setup_data(args, device):
    if args.task_name == "sinusoid":
        dataset = SinusoidNShot(
            batchsz=args.meta_batch_size,
            k_shot=args.k_shot,
            k_query=args.k_query,
            device=device,
        )
    elif args.task_name == "omniglot":
        dataset = OmniglotNShot(
            root=os.path.join(os.environ["DATA_DIR"], "omniglot"),
            n_way=args.n_way,
            k_shot=args.k_shot,
            k_query=args.k_query,
            batchsz=args.meta_batch_size,
            imgsz=28,
            device=device,
        )
    else:
        raise ValueError(f"Unknown task name {args.task_name}")
    return dataset


def setup_model(args, device):
    if args.task_name == "sinusoid":
        model = get_sinusoid_model(args)
    elif args.task_name == "omniglot":
        model = get_omniglot_model(args)
    else:
        raise ValueError(f"Unknown task name {args.task_name}")
    return model.to(device)


def setup_loss(args):
    if args.task_name == "omniglot":
        loss_fn = nn.CrossEntropyLoss()
    elif args.task_name == "sinusoid":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task name {args.task_name}")
    return loss_fn


def setup_files_and_logging(args):
    # files setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / args.task_name / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # logging setup
    logging.basicConfig(
        format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    # log config
    if args.use_wandb:
        entity = os.environ["WANDB_ENTITY"]
        project = os.environ["WANDB_PROJECT"]
        wandb.init(entity=entity, project=project, config=args, dir=str(run_dir.resolve()))
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    logging.info("Configured files and logging")


def setup_metrics(args):
    if args.task_name == "omniglot":
        metrics = {
            "acc": accuracy_score,
        }
    elif args.task_name == "sinusoid":
        metrics = {
            "mse": mean_squared_error,
        }
    else:
        raise ValueError(f"Unknown task name {args.task_name}")
    return metrics


LOSS_NAME_MAP = {
    "ce": nn.CrossEntropyLoss,
    "bcewl": nn.BCEWithLogitsLoss,
    "bce": nn.BCELoss,
    "nll": nn.NLLLoss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.AdamW,
    "adagrad": optim.Adagrad,
}
