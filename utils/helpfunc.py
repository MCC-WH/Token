import datetime
import os
import pickle
import shutil
import time
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.distributed as dist
import math
from torch import cuda
# --------------------------------------
# distributed training
# --------------------------------------


@torch.no_grad()
def extract_vectors(net, loader, ms=[1], device=torch.device('cuda')):
    net.eval()
    vecs = torch.zeros(len(loader), net.outputdim)
    if len(ms) == 1:
        for i, input in enumerate(loader):
            vecs[i, :] = net.forward_test(input.to(device)).cpu().data.squeeze()
            print('\r>>>> {}/{} done...'.format(i + 1, len(loader)), end='')
    else:
        for i, input in enumerate(loader):
            vec = torch.zeros(net.outputdim)
            for s in ms:
                if s == 1:
                    input_ = input.clone()
                else:
                    input_ = F.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
                vec += net.forward_test(input_.to(device)).cpu().data.squeeze()
            vec /= len(ms)
            vecs[i, :] = F.normalize(vec, p=2, dim=0)
            print('\r>>>> {}/{} done...'.format(i + 1, len(loader)), end='')
    return vecs


def get_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def get_data_root():
    return os.path.join(get_root(), 'data')


def get_checkpoint_root():
    return os.path.join(get_root(), 'checkpoint')


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def create_optimizer(weight_decay, model, filter_bias_and_bn=True):
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()
    return parameters


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, is_best, directory):
    if is_dist_avail_and_initialized():
        if is_main_process():
            save_checkpoints(state, is_best, directory)
    else:
        save_checkpoints(state, is_best, directory)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# --------------------------------------
# help function
# --------------------------------------
def resnet_block_dilation(block, dilation_0=None, dilation=2):
    for i in range(0, len(block)):
        # The first sub-block, containing downsample layer and stride (2,2)
        if i == 0:
            for (name, layer) in block[i].named_children():
                if 'downsample' in name:
                    layer[0].stride = (1, 1)
                if 'conv' in name:
                    if layer.stride == (2, 2):
                        layer.stride = (1, 1)
                        if dilation_0 and layer.kernel_size == (3, 3):
                            layer.dilation = (dilation_0, dilation_0)
                            layer.padding = (dilation_0, dilation_0)
        # The others sub-block, containing only simple conv2d
        else:
            for (name, layer) in block[i].named_children():
                if 'conv' in name:
                    if layer.kernel_size == (3, 3):
                        layer.dilation = (dilation, dilation)
                        layer.padding = (dilation, dilation)
    return block


def save_checkpoints(state, is_best, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.fmt = "{avg:.4f}"

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        if not math.isfinite(val):
            val = 10000.0
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum], device='cuda').float()
        dist.barrier(async_op=False)
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0]) / cuda.device_count()
        self.sum = t[1] / cuda.device_count()
        self.avg = self.sum / (self.count + 1e-6)

    def __str__(self):
        return self.fmt.format(avg=self.avg)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not math.isfinite(v):
                v = 10000.0
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable, print_freq, header=None):
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = AverageMeter()
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}', 'iter time: {time} s', 'max mem: {memory:.0f} MB'])
        else:
            log_msg = self.delimiter.join([header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}', 'iter time: {time} s'])

        MB = 1024.0 * 1024.0
        i = 0
        for obj in iterable:
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    total_memory = torch.cuda.max_memory_allocated() / MB
                    print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), memory=total_memory))
                else:
                    print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / len(iterable)))


def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_pickle(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=4)