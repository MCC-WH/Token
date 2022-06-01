import math
import os
from utils.helpfunc import get_data_root, get_checkpoint_root
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import cuda, optim
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
from config import get_args
from Dataset import GLDv2
from networks import Token
from utils import MetricLogger, create_optimizer, init_distributed_mode, is_main_process, get_rank, optimizer_to


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(preds, max(ks), dim=1, largest=True, sorted=True)
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.reshape(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks]
    return [(1.0 - x / preds.size(0)) * 100.0 for x in topks_correct]


class WarmupCos_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(math.pi * np.arange(decay_iter) / decay_iter))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        return self.lr_schedule[self.iter]

    def state_dict(self):
        state_dict = {}
        state_dict['base_lr'] = self.base_lr
        state_dict['lr_schedule'] = self.lr_schedule
        state_dict['iter'] = self.iter
        return state_dict

    def load_state_dict(self, state_dict):
        self.base_lr = state_dict['base_lr']
        self.lr_schedule = state_dict['lr_schedule']
        self.iter = state_dict['iter']


def main(args):
    init_distributed_mode(args)
    for key in vars(args):
        print(key + ":" + str(vars(args)[key]))
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    args.directory = get_checkpoint_root()
    os.makedirs(args.directory, exist_ok=True)

    if args.distributed:
        ngpus_per_node = cuda.device_count()
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print('>> batch size per node:{}'.format(args.batch_size))
        print('>> num workers per node:{}'.format(args.num_workers))

    train_dataset, val_dataset, class_num = GLDv2(csv_path=os.path.join(get_data_root(), "Google-landmark-v2/train.csv"),
                                                  clean_csv_path=os.path.join(get_data_root(), "Google-landmark-v2/train_clean.csv"),
                                                  image_dir=os.path.join(get_data_root(), "/data/Google-landmark-v2/train/"),
                                                  output_directory=get_data_root(),
                                                  imsize=args.imsize)
    args.classifier_num = class_num

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=False)
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=True)
        val_sampler = DistributedSampler(val_dataset)
        val_batch_sampler = BatchSampler(val_sampler, args.batch_size, drop_last=False)
        val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_batch_sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=False)

    model = Token(outputdim=1024, args.classifier_num).to(device)
    model_without_ddp = model

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('>> number of params:{:.2f}M'.format(n_parameters / (1024 * 1024)))

    # define optimizer
    param_dicts = create_optimizer(args.weight_decay, model_without_ddp)
    optimizer = optim.SGD(param_dicts, lr=args.base_lr * args.batch_size / 256, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True, dampening=0.0)

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model_without_ddp.load_state_dict(checkpoint, strict=False)
            optimizer.load_state_dict(checkpoint['optim'])
            optimizer_to(optimizer, device)
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))

    lr_scheduler = WarmupCos_Scheduler(optimizer=optimizer,
                                       warmup_epochs=args.warmup_epochs,
                                       warmup_lr=args.warmup_lr * args.batch_size * args.update_every / 256,
                                       num_epochs=args.num_epochs,
                                       base_lr=args.base_lr * args.batch_size * args.update_every / 256,
                                       final_lr=args.final_lr * args.batch_size * args.update_every / 256,
                                       iter_per_epoch=int(len(train_loader) / args.update_every))
    lr_scheduler.iter = max(int(len(train_loader) / args.update_every) * start_epoch, 0)

    # Start training
    metric_logger = MetricLogger(delimiter=" ")
    val_metric_logger = MetricLogger(delimiter=" ")
    print_freq = 10
    model_path = None
    Loss_logger = {'ArcFace loss': []}
    Error_Logger = {'Top1 error': [], 'Top5 error': []}
    val_Loss_logger = {'ArcFace loss': []}
    val_Error_Logger = {'Top1 error': [], 'Top5 error': []}
    min_val = 100.0

    for epoch in range(start_epoch, args.num_epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch + 1 + get_rank())
        header = '>> Train Epoch: [{}]'.format(epoch)
        optimizer.zero_grad()
        for idx, (images, targets) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            model.train()
            targets = targets.to(device, non_blocking=True)
            loss, logits = model(images.to(device, non_blocking=True), targets)
            loss.backward()
            metric_logger.meters['ArcFace loss'].update(loss.item())
            with torch.no_grad():
                desc_top1_err, desc_top5_err = topk_errors(logits, targets, [1, 5])
                metric_logger.meters['Top1 error'].update(desc_top1_err.item())
                metric_logger.meters['Top5 error'].update(desc_top5_err.item())

            if (idx + 1) % args.update_every == 0:
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                _ = lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

            if (idx + 1) % 10 == 0:
                if is_main_process():
                    Loss_logger['ArcFace loss'].append(metric_logger.meters['ArcFace loss'].avg)
                    Error_Logger['Top1 error'].append(metric_logger.meters['Top1 error'].avg)
                    Error_Logger['Top5 error'].append(metric_logger.meters['Top5 error'].avg)
                    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
                    fig.tight_layout()
                    axes = axes.flatten()
                    for (key, value) in Loss_logger.items():
                        axes[0].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                    axes[0].legend(loc='upper right', shadow=True, fontsize='medium')
                    axes[0].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                    axes[0].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                    axes[0].set_xlabel('iter')
                    axes[0].set_ylabel("loss")
                    axes[0].minorticks_on()
                    for (key, value) in Error_Logger.items():
                        axes[1].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                    axes[1].legend(loc='upper right', shadow=True, fontsize='medium')
                    axes[1].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                    axes[1].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                    axes[1].set_xlabel('iter')
                    axes[1].set_ylabel("Error rate (%)")
                    axes[1].minorticks_on()
                    plt.savefig(os.path.join(args.directory, 'training_logger.png'))
                    plt.close()

        if (epoch + 1) % args.val_epoch == 0:
            with torch.no_grad():
                # Enable eval mode
                model.eval()
                for idx, (inputs, labels) in enumerate(val_metric_logger.log_every(val_loader, print_freq, '>> Val Epoch: [{}]'.format(epoch))):
                    # Transfer the data to the current GPU device
                    inputs, labels = inputs.to(device), labels.to(device, non_blocking=True)
                    # Compute the predictions
                    loss, logits, _ = model(inputs, labels)

                    val_metric_logger.meters['ArcFace loss'].update(loss.item())
                    # Compute the errors
                    desc_top1_err, desc_top5_err = topk_errors(logits, labels, [1, 5])
                    val_metric_logger.meters['Top1 error'].update(desc_top1_err.item())
                    val_metric_logger.meters['Top5 error'].update(desc_top5_err.item())
                    if desc_top1_err.item() < min_val:
                        if is_main_process():
                            model_path = os.path.join(args.directory, 'best_checkpoint.pth')
                            torch.save({'epoch': epoch + 1, 'state_dict': model_without_ddp.state_dict(), 'optim': optimizer.state_dict()}, model_path)

                    if (idx + 1) % 10 == 0:
                        if is_main_process():
                            val_Loss_logger['ArcFace loss'].append(val_metric_logger.meters['ArcFace loss'].avg)
                            val_Error_Logger['Top1 error'].append(val_metric_logger.meters['Top1 error'].avg)
                            val_Error_Logger['Top5 error'].append(val_metric_logger.meters['Top5 error'].avg)
                            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
                            fig.tight_layout()
                            axes = axes.flatten()
                            for (key, value) in Loss_logger.items():
                                axes[0].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                            axes[0].legend(loc='upper right', shadow=True, fontsize='medium')
                            axes[0].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                            axes[0].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                            axes[0].set_xlabel('iter')
                            axes[0].set_ylabel("loss")
                            axes[0].minorticks_on()
                            for (key, value) in Error_Logger.items():
                                axes[1].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                            axes[1].legend(loc='upper right', shadow=True, fontsize='medium')
                            axes[1].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                            axes[1].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                            axes[1].set_xlabel('iter')
                            axes[1].set_ylabel("Error rate (%)")
                            axes[1].minorticks_on()
                            plt.savefig(os.path.join(args.directory, 'val_logger.png'))
                            plt.close()

        if is_main_process():
            # Save checkpoint
            model_path = os.path.join(args.directory, 'epoch{}.pth'.format(epoch + 1))
            torch.save({'epoch': epoch + 1, 'state_dict': model_without_ddp.state_dict(), 'optim': optimizer.state_dict()}, model_path)


if __name__ == "__main__":
    args = get_args()
    main(args=args)
