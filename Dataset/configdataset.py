import csv
import os
import pickle
from glob import glob
from utils import load_pickle
import numpy as np
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from utils import save_pickle
from .ImageFromList import pil_loader

ImageFile.LOAD_TRUNCATED_IMAGES = True


## ------------------------
## Testing Dataset
## ------------------------
def RoxfordAndRparis(dataset, dir_main):
    dataset = dataset.lower()

    if dataset not in ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']:
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    # loading imlist, qimlist, and gnd, in cfg as a dict
    gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
    with open(gnd_fname, 'rb') as f:
        cfg = pickle.load(f)
    cfg['gnd_fname'] = gnd_fname

    cfg['ext'] = '.jpg'
    cfg['qext'] = '.jpg'
    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = []
    for name in cfg['imlist']:
        cfg['im_fname'].append(os.path.join(cfg['dir_images'], name + '.jpg'))

    cfg['qim_fname'] = []
    for name in cfg['qimlist']:
        cfg['qim_fname'].append(os.path.join(cfg['dir_images'], name + '.jpg'))

    cfg['dataset'] = dataset

    return cfg


def get_all_image_files_and_labels(name, csv_path, image_dir):
    image_paths = glob(os.path.join(image_dir, '*/*/*/*.jpg'))
    file_ids = [os.path.basename(os.path.normpath(f))[:-4] for f in image_paths]
    if name == 'train':
        df = pd.read_csv(csv_path, dtype=str)
        df = df.set_index('id')
        labels = [int(df.loc[fid]['landmark_id']) for fid in file_ids]
    elif name == 'test':
        labels = []
    else:
        raise ValueError('Unsupported dataset split name: %s' % name)
    return image_paths, file_ids, labels


def get_clean_train_image_files_and_labels(csv_path, image_dir):
    # Load the content of the CSV file (landmark_id/label -> images).
    df = pd.read_csv(csv_path, dtype=str)
    # Create the dictionary (key = image_id, value = {label, file_id}).
    images = {}
    for _, row in df.iterrows():
        label = row['landmark_id']
        for file_id in row['images'].split(' '):
            images[file_id] = {}
            images[file_id]['label'] = label
            images[file_id]['file_id'] = file_id

    # Add the full image path to the dictionary of images.
    image_paths = glob(os.path.join(image_dir, '*/*/*/*.jpg'))
    print('>> Total image num:{}'.format(len(image_paths)))
    for image_path in image_paths:
        file_id = os.path.basename(os.path.normpath(image_path))[:-4]
        if file_id in images:
            images[file_id]['image_path'] = image_path

    # Explode the dictionary into lists (1 per image attribute).
    image_paths = []
    file_ids = []
    labels = []
    for _, value in images.items():
        image_paths.append(value['image_path'])
        file_ids.append(value['file_id'])
        labels.append(value['label'])

    # Relabel image labels to contiguous values.
    unique_labels = sorted(set(labels))
    print('>> Total number of class:{}'.format(len(unique_labels)))
    relabeling = {label: index for index, label in enumerate(unique_labels)}
    new_labels = [relabeling[label] for label in labels]
    return image_paths, file_ids, new_labels, relabeling


def write_relabeling_rules(relabeling_rules, output_directory):
    """ Write to a file the relabeling rules when the clean train dataset is used.
    Args:
        relabeling_rules: dictionary of relabeling rules applied when the clean
        train dataset is used (key = old_label, value = new_label).
    """
    relabeling_file_name = os.path.join(output_directory, 'relabeling.pkl')
    with open(relabeling_file_name, 'wb+') as relabeling_file:
        pickle.dump(relabeling_rules, relabeling_file)


def shuffle_by_columns(np_array, random_state):
    """Shuffle the columns of a 2D numpy array.
    Args:
        np_array: array to shuffle.
        random_state: numpy RandomState to be used for shuffling.
    Returns:
        The shuffled array.
    """
    columns_indices = np.arange(np_array.shape[1])
    random_state.shuffle(columns_indices)
    return np_array[:, columns_indices]


def build_train_and_validation_splits(image_paths, file_ids, labels, validation_split_size, seed):
    """Create TRAIN and VALIDATION splits containg all labels in equal proportion.
    Args:
        image_paths: list of paths to the image files in the train dataset.
        file_ids: list of image file ids in the train dataset.
        labels: list of image labels in the train dataset.
        validation_split_size: size of the VALIDATION split as a ratio of the train dataset.
        seed: seed to use for shuffling the dataset for reproducibility purposes.
    Returns:
        splits : tuple containing the TRAIN and VALIDATION splits.
    Raises:
        ValueError: if the image attributes arrays don't all have the same length, which makes the shuffling impossible.
    """
    # Ensure all image attribute arrays have the same length.
    total_images = len(file_ids)
    if not (len(image_paths) == total_images and len(labels) == total_images):
        raise ValueError('Inconsistencies between number of file_ids (%d), number of image_paths (%d) and number of labels (%d). Cannot shuffle the train dataset.' % (total_images, len(image_paths), len(labels)))

    # Stack all image attributes arrays in a single 2D array of dimensions
    # (3, number of images) and group by label the indices of datapoins in the
    # image attributes arrays. Explicitly convert label types from 'int' to 'str'
    # to avoid implicit conversion during stacking with image_paths and file_ids
    # which are 'str'.
    labels_str = [str(label) for label in labels]
    image_attrs = np.stack((image_paths, file_ids, labels_str))
    image_attrs_idx_by_label = {}
    for index, label in enumerate(labels):
        if label not in image_attrs_idx_by_label:
            image_attrs_idx_by_label[label] = []
        image_attrs_idx_by_label[label].append(index)
    total_class = len(image_attrs_idx_by_label.keys())

    # Create subsets of image attributes by label, shuffle them separately and
    # split each subset into TRAIN and VALIDATION splits based on the size of the
    # validation split.
    splits = {'val': [], 'train': []}
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
    for label, indexes in image_attrs_idx_by_label.items():
        # Create the subset for the current label.
        image_attrs_label = image_attrs[:, indexes]
        # Shuffle the current label subset.
        image_attrs_label = shuffle_by_columns(image_attrs_label, rs)
        # Split the current label subset into TRAIN and VALIDATION splits and add
        # each split to the list of all splits.
        images_per_label = image_attrs_label.shape[1]
        cutoff_idx = max(1, int(validation_split_size * images_per_label))
        splits['val'].append(image_attrs_label[:, 0:cutoff_idx])
        splits['train'].append(image_attrs_label[:, cutoff_idx:])

    # Concatenate all subsets of image attributes into TRAIN and VALIDATION splits
    # and reshuffle them again to ensure variance of labels across batches.
    validation_split = shuffle_by_columns(np.concatenate(splits['val'], axis=1), rs)
    train_split = shuffle_by_columns(np.concatenate(splits['train'], axis=1), rs)

    # Unstack the image attribute arrays in the TRAIN and VALIDATION splits and
    # convert them back to lists. Convert labels back to 'int' from 'str'
    # following the explicit type change from 'str' to 'int' for stacking.
    return ({
        'total_class': total_class,
        'image_paths': validation_split[0, :].tolist(),
        'file_ids': validation_split[1, :].tolist(),
        'labels': [int(label) for label in validation_split[2, :].tolist()]
    }, {
        'total_class': total_class,
        'image_paths': train_split[0, :].tolist(),
        'file_ids': train_split[1, :].tolist(),
        'labels': [int(label) for label in train_split[2, :].tolist()]
    })


def GLDv2_build_train_dataset(csv_path, clean_csv_path, image_dir, output_directory, generate_train_validation_splits, validation_split_size, seed):
    # Make sure the size of the VALIDATION split is inside (0, 1) if we need to
    # generate the TRAIN and VALIDATION splits.
    if generate_train_validation_splits:
        if validation_split_size <= 0 or validation_split_size >= 1:
            raise ValueError('Invalid VALIDATION split size. Expected inside (0,1) but received %f.' % validation_split_size)

    if clean_csv_path:
        # Load clean train images and labels and write the relabeling rules.
        image_paths, file_ids, labels, relabeling_rules = get_clean_train_image_files_and_labels(clean_csv_path, image_dir)
        write_relabeling_rules(relabeling_rules, output_directory)
    else:
        # Load all train images.
        image_paths, file_ids, labels = get_all_image_files_and_labels('train', csv_path, image_dir)

    if generate_train_validation_splits:
        # Generate the TRAIN and VALIDATION splits and write them to TFRecord.
        validation_split, train_split = build_train_and_validation_splits(image_paths, file_ids, labels, validation_split_size, seed)

        if clean_csv_path:
            prefix_train = os.path.join(output_directory, 'GLDv2-clean-train-split.pkl')
            prefix_val = os.path.join(output_directory, 'GLDv2-clean-val-split.pkl')
        else:
            prefix_train = os.path.join(output_directory, 'GLDv2-train-split.pkl')
            prefix_val = os.path.join(output_directory, 'GLDv2-val-split.pkl')

        save_pickle(prefix_train, train_split)
        save_pickle(prefix_val, validation_split)

    else:
        # Write to a single split, TRAIN.
        if clean_csv_path:
            prefix = os.path.join(output_directory, 'GLDv2-clean-train-all.pkl')
        else:
            prefix = os.path.join(output_directory, 'GLDv2-train-all.pkl')
        total_class = max(labels) + 1
        train = {'total_class': total_class, 'image_paths': image_paths, 'file_ids': file_ids, 'labels': labels}
        save_pickle(prefix, train)


class GLDV2Dataset(data.Dataset):
    def __init__(self, Info, transforms=None, loader=pil_loader):
        super(GLDV2Dataset, self).__init__()
        self.image_paths = Info['image_paths']
        self.labels = Info['labels']
        self.loader = loader
        self.len = len(self.image_paths)
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = self.loader(path)
        img = self.transforms(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.len


def GLDv2(csv_path, clean_csv_path, image_dir, output_directory, imsize):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # train
    pre = [transforms.RandomResizedCrop(size=imsize, scale=(0.2, 1.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8), transforms.ToTensor(), normalize]
    train_transforms = transforms.Compose(pre)

    # val
    pre = [
        transforms.Resize(int(imsize * (8 / 7)), interpolation=Image.BICUBIC),  # 224 -> 256 
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        normalize
    ]
    val_transforms = transforms.Compose(pre)

    #create your dataset here
    prefix_train = os.path.join(output_directory, 'GLDv2-clean-train-split.pkl')
    prefix_val = os.path.join(output_directory, 'GLDv2-clean-val-split.pkl')
    if os.path.exists(prefix_train) and os.path.exists(prefix_val):
        print('>> using exists dataset')
        pass
    else:
        GLDv2_build_train_dataset(csv_path, clean_csv_path, image_dir, output_directory, False, 0.2, 0)
    train_split = load_pickle(prefix_train)
    val_split = load_pickle(prefix_val)

    train_dataset = GLDV2Dataset(train_split, train_transforms)
    print('>> Total {} images for training'.format(len(train_dataset)))
    val_dataset = GLDV2Dataset(val_split, val_transforms)
    class_num = train_split['total_class']
    assert class_num == val_split['total_class']
    return train_dataset, val_dataset, class_num