import numpy as np
from utils import plot_images
from collections import defaultdict
import os

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
      In the paper, this number is set to 0.1.
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    # load dataset
    # TODO: fix dataset indexing
    class Market1501(Dataset):

        train_folder = 'train'
        val_folder = 'val'

        def __init__(self, root_dir, same_prob, train=True, transform=None):
            self.root_dir = root_dir
            self.train = train
            self.transform = transform
            self.indices = set()
            self.classes = defaultdict(set)

            assert 0.1 < same_prob < 0.6, "Resonable values pls..."
            self.same_prob = same_prob


            if self.train:
                self.data_file = datasets.ImageFolder(os.path.join(data_dir, self.train_folder), transform=trans)
                self.indices = set(range(len(self.data_file)))
                for i in range(len(self.data_file)):
                    _, target = self.data_file[i]
                    self.classes[target].add(i)

        def __len__(self):
            return len(self.data_file)

        def __getitem__(self, idx):
            sample_a, target_a = self.data_file[idx]

            if np.random.rand(1) < self.same_prob:
                candidate_idx = self.classes[target_a]
            else:
                candidate_idx = self.indices - self.classes[target_a]

            idx_b = np.random.choice(list(candidate_idx), size=1)[0]
            sample_b, target_b = self.data_file[idx_b]

            return torch.cat((sample_a, sample_b), dim=2), int(target_a == target_b)

    data_dir = './Market1501/pytorch/'

    dataset = Market1501(data_dir, 0.50)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
        data_iter = iter(sample_loader)
        images, labels = next(data_iter)
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        X = (X+1)/2
        plot_images(X, labels)

    return train_loader, valid_loader


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
