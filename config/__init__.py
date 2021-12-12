import argparse
from torch import cuda


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', metavar='EXPORT_DIR', help='destination where trained network should be saved')
    parser.add_argument('--training-dataset', default='GLDv2', help='training dataset: (default: GLDv2)')
    parser.add_argument('--imsize', default=1024, type=int, metavar='N', help='maximum size of longer image side used for training (default: 1024)')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--device', type=str, default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num-epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 100)')
    parser.add_argument('--batch-size', '-b', default=5, type=int, metavar='N', help='number of (q,p,n1,...,nN) tuples in a mini-batch (default: 5)')
    parser.add_argument('--update-every', '-u', default=1, type=int, metavar='N', help='update model weights every N batches, used to handle really large batches, ' + 'batch_size effectively becomes update_every x batch_size (default: 1)')
    parser.add_argument('--resume', default=None, type=str, metavar='FILENAME', help='name of the latest checkpoint (default: None)')

    parser.add_argument('--warmup-epochs', type=int, default=0, help='learning rate will be linearly scaled during warm up period')
    parser.add_argument('--val-epoch', type=int, default=1)
    parser.add_argument('--warmup-lr', type=float, default=0, help='Initial warmup learning rate')
    parser.add_argument('--base-lr', type=float, default=1e-6)
    parser.add_argument('--final-lr', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:29324')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clip_max_norm', type=float, default=0)
    args = parser.parse_args()
    return args
