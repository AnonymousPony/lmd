# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

import time
import datetime

mask_ratio1 = (0.15, 0.4)
mask_ratio2 = (0.4, 0.75)
current_mask_ratio, step1_final_mask_ratio = mask_ratio1
_, step2_final_mask_ratio = mask_ratio2


def cosine_schedule(steps):
    s = 0.008
    low = 0.15
    high = 0.75
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    x = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    x = x / x[0]
    y = 1 - x
    y = low + (high - low) * y
    return y.tolist()


def mr_schedule(mask_ratio, step_nums, step, linear_schedule=True):
    """
    s_mr: start_mask_ratio
    e_mr: end_mask_ratio
    step_nums: train_steps
    step: current step
    mask_ratio: return
    """
    s_mr, e_mr = mask_ratio
    linear_scale = (e_mr - s_mr) / step_nums
    mask_ratio = s_mr + linear_scale * step
    return mask_ratio


def train_one_epoch(train_num_steps: int,
                    loss_record, vae,
                    model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # cosine scheduling
    cosine_mask_ratio = cosine_schedule(int(train_num_steps * 2 / 3))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # get the training data
        samples = samples.to(device, non_blocking=True)

        # ===========================
        # compress the data using vae
        # ===========================
        with torch.no_grad():
            samples = (samples * 2.0 - 1.0)
            posterior = vae.encode(samples).latent_dist
            samples = posterior.mode()
            # ===============================
            #       Mask Ratio Schedule
            # ===============================
            step = loss_record.step

            # =======================
            #       分段线性调度
            # =======================
            # if step < train_num_steps / 6:
            #     current_mask_ratio = mr_schedule(mask_ratio1, train_num_steps / 6, step)
            # elif step < train_num_steps / 3:
            #     current_mask_ratio = step1_final_mask_ratio
            # elif step < train_num_steps * 2 / 3:
            #     current_mask_ratio = mr_schedule(mask_ratio2, train_num_steps / 3, step)
            # else:
            #     current_mask_ratio = step2_final_mask_ratio

            # =======================
            #       cosine调度
            # =======================
            if step <= train_num_steps * 2 / 3:
                current_mask_ratio = cosine_mask_ratio[step]
            else:
                current_mask_ratio = step2_final_mask_ratio

        with torch.cuda.amp.autocast():
            # loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
            loss, _, _ = model(samples, mask_ratio=current_mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        loss_record.append(loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
