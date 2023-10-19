"""https://arxiv.org/abs/1703.03400"""

import argparse
import logging
import os
import time
from pathlib import Path

import higher
import numpy as np
import torch

import wandb
from common import OPTIMIZERS, setup_data, setup_files_and_logging, setup_loss, setup_metrics, setup_model
from utils import save_checkpoint, set_seed, setup_device


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n-way", type=int, help="n way, only for classification tasks", default=20)
    argparser.add_argument("--k-shot", type=int, help="k shot for support set", default=5)
    argparser.add_argument("--k-query", type=int, help="k shot for query set", default=5)
    argparser.add_argument("--meta-batch-size", type=int, help="number of tasks per meta-update", default=16)
    argparser.add_argument("--num-epochs", type=int, help="number of epochs", default=5)
    argparser.add_argument("--meta-optimizer", type=str, help="meta optimizer", default="adam")
    argparser.add_argument("--meta-learning-rate", type=float, help="learning rate for meta optimizer", default=1e-3)
    argparser.add_argument("--inner-steps", type=int, help="number of inner gradient updates", default=5)
    argparser.add_argument(
        "--inner-learning-rate", type=float, help="step size alpha for inner gradient update", default=0.1
    )
    argparser.add_argument(
        "--task-name", type=str, help="task to train, can be `sinusoid` or `omniglot`", default="omniglot"
    )
    argparser.add_argument("--seed", type=int, help="random seed", default=42)
    argparser.add_argument("--output-dir", type=Path, help="output directory", default=os.environ["MODELS_DIR"])
    argparser.add_argument("--log-interval", type=int, help="log interval", default=10)
    argparser.add_argument("--use-wandb", action="store_true", help="use wandb for logging")
    argparser.add_argument("--run-name", type=str, help="run name", default="maml")
    args = argparser.parse_args()

    setup_files_and_logging(args)
    set_seed(args.seed)
    device = setup_device()
    dataset = setup_data(args, device)
    model = setup_model(args, device)
    meta_optimizer = OPTIMIZERS[args.meta_optimizer](model.parameters(), lr=args.meta_learning_rate)
    loss_fn = setup_loss(args)
    metrics = setup_metrics(args)

    start_time = time.time()
    logging.info(f"Starting training for {args.num_epochs} epochs")
    for epoch in range(args.num_epochs):
        meta_train_loop(model, dataset, loss_fn, meta_optimizer, metrics, epoch, args)
        meta_test_loop(model, dataset, loss_fn, epoch, metrics, args)
        save_checkpoint(
            model,
            args.output_dir / args.task_name / args.run_name / f"epoch_{epoch}.pth"
        )
    total_time = time.time() - start_time
    logging.info(f"Training completed in: {total_time:.2f}s")


def inner_loop(model, dataset, loss_fn, steps, step_size, metrics, mode="train"):
    x_spt, y_spt, x_qry, y_qry = dataset.next(mode)
    task_num = x_spt.size(0)
    inner_optimizer = torch.optim.SGD(model.parameters(), lr=step_size)
    qry_losses = []
    qry_metrics = {metric_name: [] for metric_name in metrics.keys()}
    for i in range(task_num):
        with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
            for step in range(steps):
                spt_output = fmodel(x_spt[i])
                spt_loss = loss_fn(spt_output, y_spt[i])
                diffopt.step(spt_loss)

            qry_output = fmodel(x_qry[i])
            qry_loss = loss_fn(qry_output, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu())
            for metric_name, metric_fn in metrics.items():
                qry_metrics[metric_name].append(metric_fn(qry_output.detach().cpu(), y_qry[i].cpu()))

            # Update the model's meta-parameters to optimize the query losses across all of the tasks sampled in this batch.
            # This unrolls through the gradient steps.
            if mode == "train":
                qry_loss.backward()

    return qry_losses, qry_metrics


def meta_train_loop(model, dataset, loss_fn, meta_optimizer, metrics, epoch, args):
    model.train()

    n_iter = dataset.dataset_sizes["train"] // dataset.batchsz
    for batch_idx in range(n_iter):
        start_time = time.time()
        meta_optimizer.zero_grad()
        qry_losses, qry_metrics = inner_loop(
            model, dataset, loss_fn, args.inner_steps, args.inner_learning_rate, metrics, "train"
        )
        qry_batch_loss = np.mean(qry_losses)
        qry_batch_metrics = {
            metric_name: np.mean(metric_values) for metric_name, metric_values in qry_metrics.items()
        }
        meta_optimizer.step()
        i = epoch + float(batch_idx / n_iter)
        iter_time = time.time() - start_time
        if batch_idx % args.log_interval == 0:
            loggining_str = f"[Epoch {i:.2f}] Train Loss: {qry_batch_loss:.2f} | Time: {iter_time:.2f}"
            for metric, value in qry_batch_metrics.items():
                loggining_str += f" | {metric.capitalize()}: {value:.2f}"
            logging.info(loggining_str)
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": qry_batch_loss,
                    **{f"train_{metric}": value for metric, value in qry_batch_metrics.items()},
                }
            )


def meta_test_loop(model, dataset, loss_fn, epoch, metrics, args):
    model.train()

    n_iter = dataset.dataset_sizes["test"] // dataset.batchsz
    qry_losses = []
    qry_metrics = {metric_name: [] for metric_name in metrics.keys()}
    for batch_idx in range(n_iter):
        qry_batch_losses, qry_batch_metrics = inner_loop(
            model, dataset, loss_fn, args.inner_steps, args.inner_learning_rate, metrics, "test"
        )
        qry_losses.append(qry_batch_losses)
        for metric_name, metric_values in qry_batch_metrics.items():
            qry_metrics[metric_name].append(metric_values)

    qry_loss = torch.Tensor(qry_losses).flatten().mean().item()
    qry_metrics = {
        metric_name: torch.Tensor(metric_values).flatten().mean().item() for metric_name, metric_values in qry_metrics.items()
    }
    loggining_str = f"[Epoch {epoch+1:.2f}] Test Loss: {qry_loss:.2f}"
    for metric, value in qry_metrics.items():
        loggining_str += f" | {metric.capitalize()}: {value:.2f}"
    logging.info(loggining_str)
    if args.use_wandb:
        wandb.log(
            {
                "test_loss": qry_loss,
                **{f"test_{metric}": value for metric, value in qry_metrics.items()},
            }
        )


if __name__ == "__main__":
    main()
