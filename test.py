from utils.helpfunc import get_checkpoint_root
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Dataset import ImageFromList, RoxfordAndRparis
from networks import Token, SOLAR
from utils import compute_map_and_print, extract_vectors
import os


@torch.no_grad()
def test(datasets, net, device=torch.device('cuda'), ms=[1, 2**(1 / 2), (1 / 2)**(1 / 2)], pool='local aggregation', whiten=''):
    image_size = 1024
    net.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # evaluate on test datasets
    for dataset in datasets:
        # prepare config structure for the test dataset
        cfg = RoxfordAndRparis(dataset, "./data/Roxf-Rparis/")
        images = cfg['im_fname']
        qimages = cfg['qim_fname']
        bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

        query_loader = DataLoader(ImageFromList(Image_paths=qimages, transforms=transform, imsize=image_size, bbox=bbxs), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        db_loader = DataLoader(ImageFromList(Image_paths=images, transforms=transform, imsize=image_size, bbox=None), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        # extract database and query vectors
        vecs = extract_vectors(net, db_loader, ms, device)
        qvecs = extract_vectors(net, query_loader, ms, device)

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        # search, rank, and print
        scores = np.dot(vecs, qvecs.T)
        ranks = np.argsort(-scores, axis=0)
        _, _, _ = compute_map_and_print(dataset, pool, whiten, ranks, cfg['gnd'])


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetrievalNet(81313).to(device)
    resume = os.path.join(get_checkpoint_root(), 'best_checkpoint.pth')
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
